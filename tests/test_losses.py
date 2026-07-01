import os
import numpy as np
import torch
import torch.distributed as dist
import unittest
from utils import comm
from utils.losses import AMSELoss
from parameterized import parameterized
from test_helpers import (
    setup_test_class,
)
from utils.misc_utils import (
    create_dummy_metadata,
    split_dummy_metadata,
)

from distributed.helpers import (
    compute_split_shapes,
)
from distributed.mappings import (
    split_forward_allgather_backward,
    allgather_forward_split_backward,
)


class TestLosses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_test_class(cls, True, "test_losses")

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group(None)


    @parameterized.expand(
        [
            [4, 2, 32, 720, 1440, "amse", 1e-4],
        ]
    )
    def test_amse_loss(
        self,
        B,
        T,
        C,
        H,
        W,
        loss_type,
        tolerance,
    ):

        #############################################################
        # non-distributed op
        #############################################################
        old_groups = {}
        comm_groups = list(comm._COMM_GROUPS.keys())
        for old_group in comm_groups:
            old_groups[old_group] = comm._COMM_GROUPS.pop(old_group, None)

        # create a random tensor
        pred = torch.randn(B, T, C, H, W, dtype=torch.float32, device=self.device)
        target = torch.randn(B, T, C, H, W, dtype=torch.float32, device=self.device)
        pred.requires_grad = True

        lats = torch.linspace(-90, 90, H)
        lons = torch.linspace(-180, 180, W)

        # create an AMSE loss
        metadata = create_dummy_metadata(H, W, C)
        metric = AMSELoss(
            latitudes=lats,
            longitudes=lons,
            sp_shapes=metadata["sp_shapes"],
            loss_type=loss_type,
        ).to(self.device)
        
        # forward pass
        loss = metric(pred, target)

        # backward pass
        loss.backward()
        pred_grad = pred.grad.clone()

        #############################################################
        # distributed op
        #############################################################
        # restore groups for distributed version
        for gr_name in old_groups:
            comm._COMM_GROUPS[gr_name] = old_groups[gr_name]

        sp1_shapes = compute_split_shapes(H, comm.get_size("sp1"))
        sp2_shapes = compute_split_shapes(W, comm.get_size("sp2"))
        with torch.no_grad():
            pred_local = split_forward_allgather_backward(
                pred, dim=0, comm_name="dp", shapes=None
            )
            pred_local = split_forward_allgather_backward(
                pred_local, dim=-2, comm_name="sp1", shapes=sp1_shapes
            )
            pred_local = split_forward_allgather_backward(
                pred_local, dim=-1, comm_name="sp2", shapes=sp2_shapes
            )
            target_local = split_forward_allgather_backward(
                target, dim=0, comm_name="dp", shapes=None
            )
            target_local = split_forward_allgather_backward(
                target_local, dim=-2, comm_name="sp1", shapes=sp1_shapes
            )
            target_local = split_forward_allgather_backward(
                target_local, dim=-1, comm_name="sp2", shapes=sp2_shapes
            )
        pred_local.requires_grad = True

        metadata = split_dummy_metadata(metadata, sp1_shapes, sp2_shapes)
        metric = AMSELoss(
            latitudes=lats, # global vals so don't split
            longitudes=lons,
            sp_shapes=metadata["sp_shapes"],
            loss_type=loss_type,
        ).to(self.device)
        loss_distributed = metric(pred_local, target_local)

        # backward pass
        loss_distributed.backward()
        pred_grad_local = pred_local.grad.clone()


        #############################################################
        # evaluate forward pass
        #############################################################
        dist.all_reduce(
            loss_distributed,
            op=torch.distributed.ReduceOp.AVG,
            group=comm.get_group("dp"),
        )
        err = (loss - loss_distributed).norm().item()
        if self.print_to_screen:
            print(f"final relative error of loss: {err}")

        self.assertTrue(err <= tolerance)

        #############################################################
        # evaluate backward pass
        #############################################################
        with torch.no_grad():
            pred_grad_gather = allgather_forward_split_backward(
                pred_grad_local, dim=0, shapes=None, comm_name="dp"
            ) / comm.get_size("dp")
            # sp1 and sp2 gather
            pred_grad_gather = allgather_forward_split_backward(
                pred_grad_gather, dim=-2, shapes=sp1_shapes, comm_name="sp1"
            ) 
            pred_grad_gather = allgather_forward_split_backward(
                pred_grad_gather, dim=-1, shapes=sp2_shapes, comm_name="sp2"
            ) 
            err = torch.mean(
                torch.norm(pred_grad - pred_grad_gather, p=2, dim=(1, 2, 3, 4))
                / torch.norm(pred_grad, p=2, dim=(1, 2, 3, 4))
            )
            if self.print_to_screen:
                print(f"final relative error of pred gradients in model: {err.item()}")

        self.assertTrue(err.item() <= tolerance)


if __name__ == "__main__":
    unittest.main()
