"""Swin vanilla vs Transformer Engine: copy vanilla weights into TE, compare forward/backward."""

import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from parameterized import parameterized

from distributed.helpers import (
    compute_split_shapes_for_patching,
    init_params_for_shared_weights,
)
from distributed.mappings import (
    init_ddp_model_and_reduction_hooks,
    split_forward_allgather_backward,
)
from models import swin as swin_module
from models.swin import SwinTransformer
from test_helpers import setup_test_class
from utils import comm
from utils.misc_utils import create_dummy_metadata, split_dummy_metadata


def _copy_if_same_shape(dst: torch.Tensor, src: torch.Tensor, where: str) -> None:
    if dst.shape != src.shape:
        raise AssertionError(
            f"{where}: shape mismatch {tuple(dst.shape)} vs {tuple(src.shape)}"
        )
    with torch.no_grad():
        dst.copy_(src)


def _copy_attn_vanilla_to_te(v_attn, te_attn) -> None:
    """Map vanilla WindowMultiHeadAttention weights to TE (fused bias on te.Linear)."""
    _copy_if_same_shape(te_attn.q.weight, v_attn.q.weight, "attn.q.weight")
    _copy_if_same_shape(te_attn.q.bias, v_attn.biasq.bias, "attn.q.bias")
    _copy_if_same_shape(te_attn.k.weight, v_attn.k.weight, "attn.k.weight")
    _copy_if_same_shape(te_attn.k.bias, v_attn.biask.bias, "attn.k.bias")
    _copy_if_same_shape(te_attn.v.weight, v_attn.v.weight, "attn.v.weight")
    _copy_if_same_shape(te_attn.v.bias, v_attn.biasv.bias, "attn.v.bias")
    _copy_if_same_shape(
        te_attn.q_norm.weight, v_attn.q_norm.norm.weight, "attn.q_norm.weight"
    )
    _copy_if_same_shape(
        te_attn.k_norm.weight, v_attn.k_norm.norm.weight, "attn.k_norm.weight"
    )
    _copy_if_same_shape(te_attn.proj.weight, v_attn.proj.weight, "attn.proj.weight")
    _copy_if_same_shape(te_attn.proj.bias, v_attn.biasproj.bias, "attn.proj.bias")


def _copy_block_ffn_vanilla_to_te(v_block, te_block) -> None:
    """Map vanilla norm1 / (norm2 + MLP) to TE norm1 / LayerNormMLP."""
    _copy_if_same_shape(
        te_block.norm1.weight, v_block.norm1.norm.weight, "block.norm1.weight"
    )
    mlp = v_block.mlp
    fused = te_block.rms_norm_mlp
    _copy_if_same_shape(
        fused.layer_norm_weight,
        v_block.norm2.norm.weight,
        "block.rms_norm_mlp.layer_norm",
    )
    _copy_if_same_shape(
        fused.fc1_weight, mlp.fc1.weight, "block.rms_norm_mlp.fc1_weight"
    )
    _copy_if_same_shape(fused.fc1_bias, mlp.bias1.bias, "block.rms_norm_mlp.fc1_bias")
    _copy_if_same_shape(
        fused.fc2_weight, mlp.fc2.weight, "block.rms_norm_mlp.fc2_weight"
    )
    _copy_if_same_shape(fused.fc2_bias, mlp.bias2.bias, "block.rms_norm_mlp.fc2_bias")


def copy_swin_vanilla_weights_to_te(
    model_v: SwinTransformer, model_te: SwinTransformer
) -> None:
    """
    Copy locally sharded weights from the vanilla Swin to the TE Swin (same parallel layout per rank).

    Parameter names differ (LinearLayer + BiasLayer vs te.Linear; norm2+MLP vs te.LayerNormMLP) but tensor shapes match.
    """
    model_te.patch_embed.load_state_dict(model_v.patch_embed.state_dict())

    pe_v = model_v.pos_embed
    pe_te = model_te.pos_embed
    if isinstance(pe_v, nn.Parameter):
        _copy_if_same_shape(pe_te, pe_v, "pos_embed")
    else:
        pe_te.load_state_dict(pe_v.state_dict())

    _copy_if_same_shape(model_te.head.weight, model_v.head.weight, "head.weight")

    for v_b, te_b in zip(model_v.blocks, model_te.blocks):
        _copy_attn_vanilla_to_te(v_b.attn, te_b.attn)
        _copy_block_ffn_vanilla_to_te(v_b, te_b)

    if getattr(model_v, "use_big_skip_conv", False) and hasattr(model_v, "skip_conv"):
        model_te.skip_conv.load_state_dict(model_v.skip_conv.state_dict())


class TestSwinTEMatchesVanilla(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_test_class(cls, False, "test_te")

    @classmethod
    def tearDownClass(cls):
        dist.barrier(device_ids=[cls.local_rank])
        dist.destroy_process_group(None)

    def flat_wgrad(self, model):
        chunks = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            self.assertIsNotNone(
                p.grad, msg="expected wgrad for every trainable parameter"
            )
            chunks.append(p.grad.flatten())
        vec = torch.cat(chunks)
        self.assertTrue(torch.isfinite(vec).all(), msg="non-finite wgrad")
        return vec

    @parameterized.expand(
        [
            [4, 16, 16, 4, 2, 2, 6, 8, 2, 2, True, 1e-3],
            [4, 720, 1440, 4, 9, 18, 20, 32, 8, 2, True, 1e-3],
        ]
    )
    def test_te_matches_vanilla_forward_backward(
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
        if swin_module.te is None or self.device.type != "cuda":
            self.skipTest("Needs transformer_engine and CUDA")

        temporal_context_window = 2
        meta = create_dummy_metadata(H, W, chans)
        sp1 = compute_split_shapes_for_patching(H, comm.get_size("sp1"), patch_size)
        sp2 = compute_split_shapes_for_patching(W, comm.get_size("sp2"), patch_size)
        meta = split_dummy_metadata(meta, sp1, sp2)

        model_v = SwinTransformer(
            window_size=(w1, w2),
            patch_size=patch_size,
            in_chans=chans,
            out_chans=chans,
            embed_dim=embed,
            depth=depth,
            num_heads=heads,
            temporal_context_window_size=temporal_context_window,
            coord_pos_embed=coord_pos_embed,
            use_transformer_engine=False,
            domain_metadata=meta,
        ).to(self.device)
        model_v = comm.all_model_groups_exist(model_v)

        model_te = SwinTransformer(
            window_size=(w1, w2),
            patch_size=patch_size,
            in_chans=chans,
            out_chans=chans,
            embed_dim=embed,
            depth=depth,
            num_heads=heads,
            temporal_context_window_size=temporal_context_window,
            coord_pos_embed=coord_pos_embed,
            use_transformer_engine=True,
            domain_metadata=meta,
        ).to(self.device)
        model_te = comm.all_model_groups_exist(model_te)

        init_params_for_shared_weights(model_v)
        model_v = init_ddp_model_and_reduction_hooks(
            model_v,
            device_ids=[self.local_rank],
            output_device=self.device,
            backend=self.backend,
            verbose=True,
        )
        init_params_for_shared_weights(model_te)
        model_te = init_ddp_model_and_reduction_hooks(
            model_te,
            device_ids=[self.local_rank],
            output_device=self.device,
            backend=self.backend,
            verbose=False,
        )

        copy_swin_vanilla_weights_to_te(model_v.module, model_te.module)

        inp = torch.randn(
            batch,
            temporal_context_window,
            chans,
            H,
            W,
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            inp_local = split_forward_allgather_backward(
                inp, dim=0, comm_name="dp", shapes=None
            )
            inp_local = split_forward_allgather_backward(
                inp_local, dim=-2, comm_name="sp1", shapes=sp1
            )
            inp_local = split_forward_allgather_backward(
                inp_local, dim=-1, comm_name="sp2", shapes=sp2
            )

        inp_v = inp_local.detach().clone().requires_grad_(True)
        inp_te = inp_local.detach().clone().requires_grad_(True)

        out_v = model_v(inp_v)
        out_te = model_te(inp_te)
        with torch.no_grad():
            err = torch.norm(out_v - out_te, p=2) / torch.norm(out_v, p=2)
            if self.print_to_screen:
                print(f"vanilla vs TE forward output error: {err.item()}")
        self.assertTrue(err.item() <= tolerance)

        loss_v = out_v.float().pow(2).mean()
        loss_te = out_te.float().pow(2).mean()
        loss_v.backward()
        loss_te.backward()

        with torch.no_grad():
            err = torch.norm(inp_v.grad - inp_te.grad, p=2) / torch.norm(
                inp_v.grad, p=2
            )
            if self.print_to_screen:
                print(f"vanilla vs TE input grad error: {err.item()}")
        self.assertTrue(err.item() <= tolerance)

        gv = self.flat_wgrad(model_v)
        gte = self.flat_wgrad(model_te)
        grad_diff = gv - gte
        err = grad_diff.norm().item() / gv.norm().item()
        if self.print_to_screen:
            print(f"vanilla vs TE wgrad vector error: {err}")
        self.assertTrue(err <= tolerance)


if __name__ == "__main__":
    unittest.main()
