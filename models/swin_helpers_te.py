import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Tuple, Optional
from utils.profiler_utils import profile

from models.timm_helpers import trunc_normal_

# model parallel layers
from utils import comm
from distributed.mappings import (
    roll_forward_reverseroll_backward,
    split_forward_allgather_backward,
)

import transformer_engine.pytorch as te


@torch.compile
def window_partition(x, window_size: Tuple[int, int]):
    """
    Args:
        x: (B, T, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, T, window_size, window_size, C)
    """
    B, T, H, W, C = x.shape
    x = x.view(
        B,
        T,
        H // window_size[0],
        window_size[0],
        W // window_size[1],
        window_size[1],
        C,
    )
    windows = (
        x.permute(0, 2, 4, 1, 3, 5, 6)
        .contiguous()
        .view(-1, T, window_size[0], window_size[1], C)
    )
    return windows


@torch.compile
def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
    """
    Args:
        windows: (num_windows * B, T, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, T, H, W, C)
    """
    H, W = img_size
    T = windows.shape[1]
    C = windows.shape[-1]
    ws0, ws1 = window_size

    hb = H // ws0  # number of windows along height
    wb = W // ws1  # number of windows along width
    B = windows.shape[0] // (hb * wb)

    x = windows.view(B, hb, wb, T, ws0, ws1, C)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous()
    x = x.view(B, T, H, W, C)
    return x


class WindowMultiHeadAttentionTransformerEngine(nn.Module):
    r"""This class implements window-based Multi-Head-Attention with log-spaced continuous position bias.

    Args:
        dim (int): Number of input features
        window_size (int): Window size
        num_heads (int): Number of attention heads
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int],
    ) -> None:
        super(WindowMultiHeadAttentionTransformerEngine, self).__init__()
        assert (
            dim % num_heads == 0
        ), "The number of input features (dim) are not divisible by the number of heads (num_heads)."
        self.dim: int = dim
        self.window_size: Tuple[int, int] = window_size
        self.num_heads: int = num_heads

        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        if comm.get_size("tp") > 1:
            tp_group = comm.get_group("tp")
            tp_size = comm.get_size("tp")
        else:
            tp_group = None
            tp_size = 1

        assert (
            num_heads % tp_size == 0
        ), "heads are not evenly split across TP model ranks"
        self.num_heads_local = num_heads // tp_size

        self.q = te.Linear(
            dim,
            dim,
            bias=True,
            sequence_parallel=False,
            tp_group=tp_group,
            tp_size=tp_size,
            parallel_mode="column" if tp_size > 1 else None,
            device=torch.cuda.current_device(),
        )
        self.k = te.Linear(
            dim,
            dim,
            bias=True,
            sequence_parallel=False,
            tp_group=tp_group,
            tp_size=tp_size,
            parallel_mode="column" if tp_size > 1 else None,
            device=torch.cuda.current_device(),
        )
        # QK-norm for attention stability in large models
        self.q_norm = te.RMSNorm(
            self.head_dim,
        )
        self.k_norm = te.RMSNorm(
            self.head_dim,
        )
        self.q_norm.weight.comm_metadata = {
            "sharded": [None],
            "shared": ["tp-sp1-sp2"],
            "reduce": ["tp-sp1-sp2"],
        }
        self.k_norm.weight.comm_metadata = {
            "sharded": [None],
            "shared": ["tp-sp1-sp2"],
            "reduce": ["tp-sp1-sp2"],
        }
        self.v = te.Linear(
            dim,
            dim,
            bias=True,
            sequence_parallel=False,
            tp_group=tp_group,
            tp_size=tp_size,
            parallel_mode="column" if tp_size > 1 else None,
            device=torch.cuda.current_device(),
        )
        self.proj = te.Linear(
            dim,
            dim,
            bias=True,
            sequence_parallel=False,
            tp_group=tp_group,
            tp_size=tp_size,
            parallel_mode="row" if tp_size > 1 else None,
            device=torch.cuda.current_device(),
        )

        # mark weights
        self.q.weight.comm_metadata = {
            "sharded": ["tp", None],
            "shared": ["sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        self.q.bias.comm_metadata = {
            "sharded": ["tp"],
            "shared": ["sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        self.k.weight.comm_metadata = {
            "sharded": ["tp", None],
            "shared": ["sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        self.k.bias.comm_metadata = {
            "sharded": ["tp"],
            "shared": ["sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        self.v.weight.comm_metadata = {
            "sharded": ["tp", None],
            "shared": ["sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        self.v.bias.comm_metadata = {
            "sharded": ["tp"],
            "shared": ["sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        self.proj.weight.comm_metadata = {
            "sharded": [None, "tp"],
            "shared": ["sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        self.proj.bias.comm_metadata = {
            "sharded": [],
            "shared": ["tp-sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }

    def _qk_and_qknorm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Bw, L, C = x.shape

        q = (
            self.q(x)
            .reshape(Bw, L, self.num_heads_local, self.head_dim)
        )
        k = (
            self.k(x)
            .reshape(Bw, L, self.num_heads_local, self.head_dim)
        )

        # QK-norm: normalize Q and K before attention to bound logits
        q = self.q_norm(q).permute(0, 2, 1, 3)
        k = self.k_norm(k).permute(0, 2, 1, 3)

        return q, k

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: shape (B*num_windows, T*window_area, C)
            mask: shape (num_windows, T*window_area, T*window_area)
        """
        Bw, L, C = x.shape

        x = x.contiguous()

        # q, k = checkpoint(self._qk_and_qknorm, x, use_reentrant=False)
        q, k = self._qk_and_qknorm(x)

        v = (
            self.v(x)
            .reshape(Bw, L, self.num_heads_local, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        if mask is not None:
            num_win: int = mask.shape[0]
            # mask needs to be broadcastable to attention scores (Bw, H, L, L)
            mask = (
                mask.unsqueeze(0)
                .expand(Bw // num_win, num_win, L, L)
                .reshape(Bw, L, L)
                .unsqueeze(1)
            )

        # Use scaled_dot_product_attention
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)

        x = x.transpose(1, 2).reshape(Bw, L, -1)
        x = self.proj(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        feat_size,
        window_size,
        temporal_context_window_size,
        shift_size=(0, 0),
        mlp_ratio=4.0,
        drop=0.0,
        act_layer=nn.GELU,
        sp1_shapes=None,
        sp2_shapes=None,
    ):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.sp1_shapes = sp1_shapes
        self.sp2_shapes = sp2_shapes
        self.feat_size = feat_size
        self.window_size = window_size
        self.window_area = self.window_size[0] * self.window_size[1]
        self.temporal_context_window_size = temporal_context_window_size
        self.shift_size = shift_size

        if comm.get_size("tp") > 1:
            tp_group = comm.get_group("tp")
            tp_size = comm.get_size("tp")
        else:
            tp_group = None
            tp_size = 1

        self.norm1 = te.RMSNorm(
            dim,
        )
        self.norm1.weight.comm_metadata = {
            "sharded": [None],
            "shared": ["tp-sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }

        attention_layer = WindowMultiHeadAttentionTransformerEngine

        self.attn = attention_layer(
            dim,
            num_heads=num_heads,
            window_size=window_size,
        )
        self.rms_norm_mlp = te.LayerNormMLP(
            hidden_size=dim,
            ffn_hidden_size=mlp_hidden_dim,
            normalization="RMSNorm",
            activation="gelu",
            init_method=self._init_weights_for_te,
            output_layer_init_method=self._init_weights_for_te,
            sequence_parallel=False,
            set_parallel_mode=True,
            tp_group=tp_group,
            tp_size=tp_size,
            device=torch.cuda.current_device(),
        )

        # set metadata for sp
        self.rms_norm_mlp.layer_norm_weight.comm_metadata = {
            "sharded": [None],
            "shared": ["tp-sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        self.rms_norm_mlp.fc1_weight.comm_metadata = {
            "sharded": ["tp", None],
            "shared": ["sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        self.rms_norm_mlp.fc1_bias.comm_metadata = {
            "sharded": ["tp"],
            "shared": ["sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        self.rms_norm_mlp.fc2_weight.comm_metadata = {
            "sharded": [None, "tp"],
            "shared": ["sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }
        self.rms_norm_mlp.fc2_bias.comm_metadata = {
            "sharded": [],
            "shared": ["tp-sp1-sp2"],
            "reduce": ["sp1-sp2"],
        }

        self._make_attention_mask(T=self.temporal_context_window_size)

    def _init_weights_for_te(self, weight):
        trunc_normal_(weight, std=0.02)

    def _make_attention_mask(self, T: int, causal: bool = False) -> None:
        """Generates the spatial + temporal attention mask used in shifted window self-attention."""

        H, W = self.feat_size
        H_global = sum(self.sp1_shapes)
        W_global = sum(self.sp2_shapes)
        # TODO: just create the global mask for simplicity and split
        # but later create local masks
        img_mask = torch.zeros((1, 1, H_global, W_global, 1))
        cnt = 0

        # temporal mask
        if causal:
            # upper-triangular mask
            temporal = torch.triu(torch.ones(T, T), diagonal=1)
            temporal = temporal.masked_fill(temporal == 1, float("-100.0")).masked_fill(
                temporal == 0, float(0.0)
            )  # (T, T)
            mask_shape = (T, T)
        else:
            temporal = torch.zeros(T, T)
            # mask shape is the same here but could be diff if doing cross-attention
            # so just keeping the var here for now
            mask_shape = (T, T)

        temporal = temporal.unsqueeze(-1).unsqueeze(-1)  # mask_shape + (1, 1)
        temporal = temporal.expand(
            mask_shape + (self.window_area, self.window_area)
        )  # mask_shape + (window_area, window_area)
        temporal = temporal.unsqueeze(0)  # (1, *mask_shape, window_area, window_area)

        if any(self.shift_size):
            for h in (slice(0, -self.window_size[0]), slice(-self.shift_size[0], None)):
                img_mask[:, :, h, :, :] = cnt
                cnt += 1

            with torch.no_grad():
                img_mask = split_forward_allgather_backward(
                    img_mask, dim=-3, comm_name="sp1", shapes=self.sp1_shapes
                )
                img_mask = split_forward_allgather_backward(
                    img_mask, dim=-2, comm_name="sp2", shapes=self.sp2_shapes
                )
            mask_windows = window_partition(
                img_mask, self.window_size
            )  # (num_windows, 1, wH, wW, 1)
            mask_windows = mask_windows.view(
                -1, self.window_area
            )  # (num_windows, window_area)

            # spatial mask (same as original swin)
            spatial_attn = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
                2
            )  # (num_windows, window_area, window_area)
            spatial_attn = spatial_attn.masked_fill(
                spatial_attn != 0, float(-100.0)
            ).masked_fill(spatial_attn == 0, float(0.0))

            spatial = spatial_attn.unsqueeze(1).unsqueeze(
                2
            )  # (num_windows, 1, 1, window_area, window_area)
            spatial = spatial.expand(
                -1, *mask_shape, -1, -1
            )  # (num_windows, *mask_shape, window_area, window_area)
            combined = spatial + temporal
        else:
            num_windows = H // self.window_size[0] * W // self.window_size[1]
            combined = temporal.expand(
                num_windows, *mask_shape, self.window_area, self.window_area
            )

        # in the end, it should be (num_windows, L, L) with L = T * window_area
        combined = combined.permute(0, 1, 3, 2, 4).reshape(
            -1, mask_shape[0] * self.window_area, mask_shape[1] * self.window_area
        )
        self.register_buffer("attn_mask", combined, persistent=False)

    @profile(name="shifted_wmha")
    def _shifted_window_attn(self, x):
        B, T, H, W, C = x.shape

        # cyclic shift
        shifts = tuple(-sh for sh in self.shift_size)
        x = roll_forward_reverseroll_backward(
            x, shifts=shifts, dims=(-3, -2), comm_names=["sp1", "sp2"]
        )

        # partition windows
        x_windows = window_partition(
            x, self.window_size
        )  # num_windows * B, T, window_size, window_size, C
        x_windows = x_windows.view(-1, T * self.window_size[0] * self.window_size[1], C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # num_windows * B, T * window_size * window_size, C

        # merge windows
        attn_windows = attn_windows.view(
            -1, T, self.window_size[0], self.window_size[1], C
        )
        x = window_reverse(
            attn_windows, self.window_size, self.feat_size
        )  # B T H' W' C

        # reverse cyclic shift
        x = roll_forward_reverseroll_backward(
            x, shifts=self.shift_size, dims=(-3, -2), comm_names=["sp1", "sp2"]
        )

        return x

    def forward(
        self, x: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of the shape [B, T, H, W, C]

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, T, H, W, C]
        """
        skip = x
        x = self.norm1(x)
        x = self._shifted_window_attn(x)
        x = x + skip

        B, T, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        skip = x
        x = self.rms_norm_mlp(x)
        x = x + skip
        x = x.reshape(B, T, H, W, C)

        return x
