import torch
from torch import nn as nn
import torch.nn.functional as F
from models.timm_helpers import trunc_normal_
from utils import comm


class LinearLayer(nn.Module):
    """
    Wrapper for nn.Linear in distributed mode
    mark params for sharing/reduction

     Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        comm_metadata (Dict[str, List[str]]): Dictionary of communication metadata
            "sharded": Communication groups where the weights are sharded for each dim
            "shared": Communication groups where the weights are shared
            "reduce": Communication groups where the weights are reduced
            Default is {"sharded": [None, None], "shared": [], "reduce": []}
    """

    def __init__(
        self,
        in_features,
        out_features,
        comm_metadata={
            "sharded": [None, None],
            "shared": [],
            "reduce": [],
        },
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight.comm_metadata = comm_metadata
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.weight, std=0.02)

    def forward(self, x):
        return F.linear(x, self.weight, bias=None)


class BiasLayer(nn.Module):
    """
    Wrapper for bias in distributed mode
    mark params for sharing/reduction

    Args:
        dim (int): Dimension of the bias
        comm_metadata (Dict[str, List[str]]): Dictionary of communication metadata
            "sharded": Communication groups where the weights are sharded for each dim
            "shared": Communication groups where the weights are shared
            "reduce": Communication groups where the weights are reduced
            Default is {"sharded": [None], "shared": [], "reduce": []}

    note, assumed that bias will always need its wgrads reduced in sp group
    """

    def __init__(self, dim, comm_metadata={
            "sharded": [None],
            "shared": [],
            "reduce": [],
        },
    ):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.bias.comm_metadata = comm_metadata

    def forward(self, x):
        return x + self.bias


class NormLayer(nn.Module):
    """
    Wrapper for normalization layers in distributed mode
    mark params for sharing/reduction
    """

    def __init__(
        self,
        normalized_shape,
        elementwise_affine=True,
        norm_type="layer_norm",  # or "rms_norm"
        comm_metadata={
            "sharded": [None],
            "shared": [],
            "reduce": [],
        },
    ):
        super(NormLayer, self).__init__()

        if norm_type == "layer_norm":
            bias = True
            self.norm = nn.LayerNorm(
                normalized_shape,
                elementwise_affine=elementwise_affine,
                bias=bias,
            )
        elif norm_type == "rms_norm":
            bias = False
            self.norm = nn.RMSNorm(
                normalized_shape,
                elementwise_affine=elementwise_affine,
            )
        else:
            raise ValueError(f"Invalid norm type: {norm_type}")

        if elementwise_affine:
            # affine weights need additional allreduce and are shared
            # across all groups; handle this within DDP since its wgrads
            self.norm.weight.comm_metadata = comm_metadata
            if bias:
                self.norm.bias.comm_metadata = comm_metadata

    @torch.compile
    def forward(self, x):
        return self.norm(x)


class PatchEmbedLayer(nn.Module):
    """Wrapper for distributed
    Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        sp1_shapes=None,
        sp2_shapes=None,
        comm_metadata={
            "sharded": [],
            "shared": ["tp-sp1-sp2"],
            "reduce": ["sp1-sp2"],
        },
    ):
        super().__init__()

        self.proj = Conv2DLayer(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
            comm_metadata=comm_metadata,
        )

        # get the local patch shapes
        self.sp1_patch_shapes = [s // patch_size for s in sp1_shapes]
        self.h = self.sp1_patch_shapes[comm.get_rank("sp1")]
        self.sp2_patch_shapes = [s // patch_size for s in sp2_shapes]
        self.w = self.sp2_patch_shapes[comm.get_rank("sp2")]
        self.num_patches = self.h * self.w

    def forward(self, x):
        return self.proj(x)  # B, C, H, W

class Conv2DLayer(nn.Module):
    """Wrapper for distributed
    Convolutional 2D Layer"""

    def __init__(self, in_chans, out_chans, kernel_size, stride=1, bias=True, comm_metadata={
        "sharded": [],
        "shared": ["tp-sp1-sp2"],
        "reduce": ["sp1-sp2"],
    }): 
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride, bias=bias)
        self.conv.weight.comm_metadata = comm_metadata
        if bias:
            self.conv.bias.comm_metadata = comm_metadata

    def forward(self, x):
        return self.conv(x)