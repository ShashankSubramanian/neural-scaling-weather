import os
import torch
import torch.distributed as dist
import unittest
from utils import comm

from utils.losses import WeightedLoss, AMSELoss

from models.swin import SwinTransformer
from parameterized import parameterized

from distributed.helpers import (
    init_params_for_shared_weights,
    compute_split_shapes_for_patching,
)
from distributed.mappings import (
    split_forward_allgather_backward,
    allgather_forward_split_backward,
    init_ddp_model_and_reduction_hooks,
)
from test_helpers import (
    setup_test_class,
)
from utils.misc_utils import (
    create_dummy_metadata,
    split_dummy_metadata,
)

# from ic_logger import setup_ic_logger


class TestDistributed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_test_class(cls, True, "test_all")
        # setup_ic_logger("test_all")

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group(None)

    def _get_flattened_grads(self, model):
        """Get flattened gradients from a model"""
        grads = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        return torch.cat(grads) if grads else torch.tensor([], device=self.device)

    def _get_reconstructed_grads(self, model):
        """Get reconstructed gradients from a distributed model, handling shared parameters"""
        all_grads = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                # check which param is sharded
                is_sharded = len(param.comm_metadata["sharded"]) > 0

                if not is_sharded:
                    # just append since the grad is shared as well
                    all_grads.append(param.grad.view(-1))
                else:
                    # sharded params, so gather them
                    current_grad = param.grad
                    sharded_info = param.comm_metadata["sharded"]

                    # Process each sharding dimension
                    for i, sharded_comm in enumerate(sharded_info):
                        if sharded_comm is not None:
                            sharded_comm_size = comm.get_size(sharded_comm)
                            gathered_grads = [
                                torch.zeros_like(current_grad)
                                for _ in range(sharded_comm_size)
                            ]
                            dist.all_gather(
                                gathered_grads,
                                current_grad,
                                group=comm.get_group(sharded_comm),
                            )
                            gather_dim = i
                            current_grad = torch.cat(gathered_grads, dim=gather_dim)

                    # after all communication groups are done, flatten the final result
                    all_grads.append(current_grad.view(-1))

        return (
            torch.cat(all_grads) if all_grads else torch.tensor([], device=self.device)
        )

    def _copy_param_weights(self, param_src, param_dst):
        """Copy weights from source parameter to destination parameter using comm_metadata"""
        if (
            not hasattr(param_dst, "comm_metadata")
            or len(param_dst.comm_metadata["sharded"]) == 0
        ):
            # not sharded, copy directly
            param_dst.copy_(param_src)
            return

        with torch.no_grad():
            sharded_info = param_dst.comm_metadata["sharded"]
            current_tensor = param_src.clone()

            for i, sharded_comm in enumerate(sharded_info):
                if sharded_comm is not None:
                    comm_size = comm.get_size(sharded_comm)
                    rank = comm.get_rank(sharded_comm)
                    if sharded_comm in ["sp1", "sp2"]:
                        # use context_patch_shapes since shapes may not be uniform
                        start = sum(self.local_patch_shapes[sharded_comm][:rank])
                        local_size = self.local_patch_shapes[sharded_comm][rank]
                        end = start + local_size
                    else:
                        # local sizes are uniform
                        dim_size = current_tensor.shape[i]
                        local_size = dim_size // comm_size
                        start = rank * local_size
                        end = start + local_size

                    # create slice indices
                    slice_indices = [slice(None)] * len(current_tensor.shape)
                    slice_indices[i] = slice(start, end)

                    # extract the local slice
                    current_tensor = current_tensor[slice_indices]

            # copy the final sharded tensor
            param_dst.copy_(current_tensor)

    def _copy_model_weights(self, model_src, model_dst):
        """Copy all weights from source model to destination model using comm_metadata"""
        with torch.no_grad():
            # Get all parameters from both models
            src_params = dict(model_src.named_parameters())
            dst_params = dict(model_dst.named_parameters())

            # Copy each parameter
            for name, dst_param in dst_params.items():
                if name in src_params:
                    src_param = src_params[name]
                    self._copy_param_weights(src_param, dst_param)
                else:
                    print(f"Warning: Parameter {name} not found in source model")

    def _copy_mlp_weights(self, mlp_layer, mlp_layer_distributed):
        """copy the weights, bias of mlp into the correct shard of mlp_dist"""
        self._copy_model_weights(mlp_layer, mlp_layer_distributed)

    def _copy_block_weights(self, block, block_distributed):
        """copy the weights of block into the correct shard of block_dist"""
        self._copy_model_weights(block, block_distributed)

    def _copy_swin_weights(self, model, model_distributed):
        """Copy all weights from a SwinTransformer model to its distributed version"""
        # get local patch shapes in sp1 and sp2 (for pos_embed)
        self.local_patch_shapes = {
            "sp1": model_distributed.patch_embed.sp1_patch_shapes,
            "sp2": model_distributed.patch_embed.sp2_patch_shapes,
        }
        self._copy_model_weights(model, model_distributed)

    # tests to run with input parameterization
    # inputs are batch, H, W, embed, num_heads, window_size1, window_size2, tolerance,
    # head_type, number of channels, depth, coord_pos_embed
    @parameterized.expand(
        [
            [4, 16, 16, 4, 2, 2, 6, 8, 2, 2, False, 1e-4],
            [4, 16, 16, 4, 2, 2, 6, 8, 2, 2, True, 1e-4],
            [4, 720, 1440, 4, 9, 18, 20, 32, 8, 2, False, 1e-4],
            [4, 720, 1440, 4, 9, 18, 20, 32, 8, 2, True, 1e-4],
        ]
    )
    def test_distributed_model(
        self,
        batch,
        H,
        W,
        patch_size,
        w1,
        w2,
        chans,
        embed,
        heads,
        depth,
        coord_pos_embed,
        tolerance,
    ):
        #############################################################
        # non-distributed op
        #############################################################
        # temporarily remove tp and tp-sp groups for non-distributed version
        old_groups = {}
        comm_groups = list(comm._COMM_GROUPS.keys())
        for old_group in comm_groups:
            old_groups[old_group] = comm._COMM_GROUPS.pop(old_group, None)
        temporal_context_window = 2
        # single gpu domain
        metadata = create_dummy_metadata(H, W, chans)
        model = SwinTransformer(
            window_size=(w1, w2),
            patch_size=patch_size,
            in_chans=chans,
            out_chans=chans,
            embed_dim=embed,
            depth=depth,
            num_heads=heads,
            temporal_context_window_size=temporal_context_window,
            coord_pos_embed=coord_pos_embed,
            domain_metadata=metadata,
        ).to(self.device)
        model.zero_grad(set_to_none=True)

        # create tensor in BHWC format
        inp = torch.randn(
            (batch, temporal_context_window, chans, H, W),
            dtype=torch.float32,
            device=self.device,
        )
        inp.requires_grad = True

        # forward pass
        out = model(inp)
        target = torch.randn_like(out)
        loss_func = WeightedLoss(area_weighting=True, weights=metadata["area_weights"]).to(self.device)
        # loss_func = AMSELoss(
        #     latitudes=torch.linspace(-90, 90, H),
        #     longitudes=torch.linspace(-180, 180, W),
        #     sp_shapes=metadata["sp_shapes"],
        # ).to(self.device)
        loss = loss_func(out, target)
        loss.backward()

        non_dist_grads = self._get_flattened_grads(model)

        inp_grad = inp.grad.clone()

        #############################################################
        # distributed op
        #############################################################
        # restore groups for distributed version
        for group_name, group in old_groups.items():
            if group is not None:
                comm._COMM_GROUPS[group_name] = group

        # compute split shapes : split the input image based on what the patch size will be
        sp1_shapes = compute_split_shapes_for_patching(
            H, comm.get_size("sp1"), patch_size
        )
        sp2_shapes = compute_split_shapes_for_patching(
            W, comm.get_size("sp2"), patch_size
        )

        # split domain data
        metadata = split_dummy_metadata(metadata, sp1_shapes, sp2_shapes)

        model_distributed = SwinTransformer(
            window_size=(w1, w2),
            patch_size=patch_size,
            in_chans=chans,
            out_chans=chans,
            embed_dim=embed,
            depth=depth,
            num_heads=heads,
            temporal_context_window_size=temporal_context_window,
            coord_pos_embed=coord_pos_embed,
            domain_metadata=metadata,
        ).to(self.device)
        model_distributed.zero_grad(set_to_none=True)

        # do some comm checks to make sure all groups are initialized
        model_distributed = comm.all_model_groups_exist(model_distributed)

        # wrap in DDP
        init_params_for_shared_weights(model_distributed)  # mark shared params
        model_distributed = init_ddp_model_and_reduction_hooks(
            model_distributed,
            device_ids=[self.local_rank],
            output_device=self.device,
            backend=self.backend,
            verbose=True,
        )

        # sync the weights
        self._copy_swin_weights(model, model_distributed.module)

        # forward pass
        with torch.no_grad():
            # dp split
            inp_local = split_forward_allgather_backward(
                inp, dim=0, comm_name="dp", shapes=None
            )
            target_local = split_forward_allgather_backward(
                target, dim=0, comm_name="dp", shapes=None
            )
            # sp1 and sp2 split
            inp_local = split_forward_allgather_backward(
                inp_local, dim=-2, comm_name="sp1", shapes=sp1_shapes
            )
            inp_local = split_forward_allgather_backward(
                inp_local, dim=-1, comm_name="sp2", shapes=sp2_shapes
            )
            target_local = split_forward_allgather_backward(
                target_local, dim=-2, comm_name="sp1", shapes=sp1_shapes
            )
            target_local = split_forward_allgather_backward(
                target_local, dim=-1, comm_name="sp2", shapes=sp2_shapes
            )
        inp_local.requires_grad = True
        out_local = model_distributed(inp_local)

        loss_func = WeightedLoss(
            area_weighting=True, weights=metadata["area_weights"]
        ).to(self.device)
        # loss_func = AMSELoss(
            # latitudes=torch.linspace(-90, 90, H),
            # longitudes=torch.linspace(-180, 180, W),
            # sp_shapes=metadata["sp_shapes"],
        # ).to(self.device)
        loss = loss_func(out_local, target_local)
        loss.backward()
        inp_grad_local = inp_local.grad.clone()

        #############################################################
        # evaluate forward pass
        #############################################################
        with torch.no_grad():
            # compute error over spatial dimensions
            out_gather = allgather_forward_split_backward(
                out_local, dim=0, shapes=None, comm_name="dp"
            )
            # sp1 and sp2 gather
            out_gather = allgather_forward_split_backward(
                out_gather, dim=-2, shapes=sp1_shapes, comm_name="sp1"
            )
            out_gather = allgather_forward_split_backward(
                out_gather, dim=-1, shapes=sp2_shapes, comm_name="sp2"
            )
            err = torch.mean(
                torch.norm(out - out_gather, p=2, dim=(1, 2, 3, 4))
                / torch.norm(out, p=2, dim=(1, 2, 3, 4))
            )
            if self.print_to_screen:
                print(f"final relative error of output in model: {err.item()}")
        self.assertTrue(err.item() <= tolerance)

        #############################################################
        # evaluate backward pass
        #############################################################
        with torch.no_grad():
            inp_grad_gather = allgather_forward_split_backward(
                inp_grad_local, dim=0, shapes=None, comm_name="dp"
            ) / comm.get_size("dp")
            # sp1 and sp2 gather
            inp_grad_gather = allgather_forward_split_backward(
                inp_grad_gather, dim=-2, shapes=sp1_shapes, comm_name="sp1"
            )
            inp_grad_gather = allgather_forward_split_backward(
                inp_grad_gather, dim=-1, shapes=sp2_shapes, comm_name="sp2"
            )
            err = torch.mean(
                torch.norm(inp_grad - inp_grad_gather, p=2, dim=(1, 2, 3, 4))
                / torch.norm(inp_grad, p=2, dim=(1, 2, 3, 4))
            )
            if self.print_to_screen:
                print(f"final relative error of input gradients in model: {err.item()}")

        self.assertTrue(err.item() <= tolerance)

        #############################################################
        # evaluate wgrads
        #############################################################
        dist_grads_full = self._get_reconstructed_grads(model_distributed.module)
        # Compare with the non-distributed gradients
        grad_diff = non_dist_grads - dist_grads_full
        err = grad_diff.norm().item() / non_dist_grads.norm().item()
        if self.print_to_screen:
            print(f"final relative error of weight gradients: {err}")

        self.assertTrue(err <= tolerance)


if __name__ == "__main__":
    unittest.main()
