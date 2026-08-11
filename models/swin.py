import torch
import torch.nn as nn
import math
from models.timm_helpers import trunc_normal_

from models.swin_helpers import Block as BlockDefault

try:
    import transformer_engine.pytorch as te
except ImportError:
    te = None

if te is not None:
    from models.swin_helpers_te import Block as BlockTE
else:
    BlockTE = None

from omegaconf import ListConfig
from models.layer_helpers import (
    LinearLayer,
    PatchEmbedLayer,
    Conv2DLayer,
)
from utils.profiler_utils import profile

# model parallel layers
from utils import comm
from utils.misc_utils import create_dummy_metadata
from distributed.helpers import compute_split_shapes
from distributed.mappings import (
    split_forward_allgather_backward,
    allgather_forward_split_backward,
)


class SwinTransformer(nn.Module):
    """
    Swin Transformer implementation.
    Assumes input is a 5D tensor of shape (batch_size, temporal_context_window_size, in_channels, height, width).
    Output is a 5D tensor of shape (batch_size, temporal_context_window_size, out_channels, height, width).

    Args:
        img_size (Tuple[int, int], optional): Dimensions of the input image as [height, width]. Default: (721, 1440).
        temporal_context_window_size (int, optional): Size of the temporal context window. Default: 2.
        window_size (Tuple[int, int], optional): Size of windows for attention. Default: (9, 9).
        patch_size (int, optional): Size of each image patch (both height and width). Default: 16.
        in_chans (int, optional): Number of input image channels. Default: 3.
        out_chans (int, optional): Number of output channels (e.g., for reconstruction tasks). Default: 3.
        embed_dim (int, optional): Dimension of the patch embeddings. Default: 768.
        depth (int, optional): Number of transformer blocks (layers). Default: 12.
        num_heads (int, optional): Number of attention heads in the multi-head self-attention mechanism. Default: 12.
        mlp_ratio (float, optional): Expansion ratio for the MLP layers in each transformer block. Default: 4.0.
        drop_rate (float, optional): Dropout rate applied to embeddings and MLP layers. Default: 0.0.
        attn_drop_rate (float, optional): Dropout rate applied to attention weights. Default: 0.0.
        drop_path_rate (float, optional): Stochastic depth rate for residual connections. Default: 0.0.
        coord_pos_embed (bool, optional): If True, use coordinate-based positional embedding instead of learnable parameters. Default: False.
        domain_metadata (dict, optional): Metadata about the domain. Default: None.
        **kwargs: Additional arguments for flexibility or future extensions.
    """

    @classmethod
    def instantiate_from_cfg(cls, cfg, domain_metadata):
        if isinstance(cfg.model.window_size, ListConfig):
            window_size = tuple(cfg.model.window_size)
        else:
            window_size = (cfg.model.window_size, cfg.model.window_size)
        # if we have invariants, we need to add them to the input
        extra_inputs = len(cfg.data.invariants) if cfg.data.invariants else 0
        return cls(
            patch_size=cfg.model.patch_size,
            window_size=window_size,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            temporal_context_window_size=cfg.train.temporal_context_window,
            in_chans=domain_metadata["n_channels"] + extra_inputs,
            out_chans=domain_metadata["n_channels"],
            embed_dim=cfg.model.embed_dim,
            mlp_ratio=cfg.model.mlp_ratio,
            drop_rate=cfg.model.dropout,
            coord_pos_embed=getattr(cfg.model, "coord_pos_embed", True),
            use_transformer_engine=getattr(
                cfg.parallelism, "use_transformer_engine", True
            ),
            domain_metadata=domain_metadata,
        )

    def __init__(
        self,
        temporal_context_window_size=1,
        window_size=(9, 9),
        patch_size=16,
        in_chans=3,
        out_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        coord_pos_embed=True,
        domain_metadata=None,
        use_transformer_engine=True,
        use_big_skip_conv=False,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.temporal_context_window_size = temporal_context_window_size
        self.window_size = window_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_rate = drop_rate
        self.coord_pos_embed = coord_pos_embed
        self.domain_metadata = domain_metadata
        self.use_transformer_engine = use_transformer_engine
        self.use_big_skip_conv = use_big_skip_conv

        # patch embedding layer
        # weights are shared in tp-sp, but dgrads are only sharded in sp (shared in tp)
        self.patch_embed = PatchEmbedLayer(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
            sp1_shapes=self.domain_metadata["sp_shapes"][0],
            sp2_shapes=self.domain_metadata["sp_shapes"][1],
            comm_metadata={
                "sharded": [],
                "shared": ["tp-sp1-sp2"],
                "reduce": ["sp1-sp2"],
            },
        )

        if self.coord_pos_embed:
            self._init_coords()
            # weights are shared in tp-sp, but the dgrads are only sharded in sp (shared in tp)
            self.pos_embed = PatchEmbedLayer(
                patch_size=patch_size,
                in_chans=self.coords.shape[-1],
                embed_dim=self.embed_dim,
                sp1_shapes=self.domain_metadata["sp_shapes"][0],
                sp2_shapes=self.domain_metadata["sp_shapes"][1],
                comm_metadata={
                    "sharded": [],
                    "shared": ["tp-sp1-sp2"],
                    "reduce": ["sp1-sp2"],
                },
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    self.temporal_context_window_size,
                    self.patch_embed.h,
                    self.patch_embed.w,
                    self.embed_dim,
                )
            )
            trunc_normal_(self.pos_embed, std=0.02)
            # weights are shared in tp only, wgrads and dgrads are same
            self.pos_embed.comm_metadata = {
                "sharded": [None, None, "sp1", "sp2", None],
                "shared": ["tp"],
                "reduce": [],
            }

        if self.use_transformer_engine and te is None:
            if comm.get_world_rank() == 0:
                print(
                    "SwinTransformer: use_transformer_engine=True but Transformer Engine "
                    "is not available (import failed); using BlockDefault."
                )

        block_cls = (
            BlockTE
            if (te is not None and self.use_transformer_engine)
            else BlockDefault
        )

        self.blocks = nn.ModuleList(
            [
                block_cls(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    feat_size=(self.patch_embed.h, self.patch_embed.w),
                    window_size=window_size,
                    temporal_context_window_size=temporal_context_window_size,
                    shift_size=tuple(
                        [0 if ((i % 2) == 0) else w // 2 for w in window_size]
                    ),
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    sp1_shapes=self.patch_embed.sp1_patch_shapes,
                    sp2_shapes=self.patch_embed.sp2_patch_shapes,
                )
                for i in range(depth)
            ]
        )

        self.out_size = self.out_chans * self.patch_size * self.patch_size
        # weights are shared in tp-sp, but dgrads are also shared in tp-sp, so no reductions needed
        self.head = LinearLayer(
            embed_dim,
            self.out_size,
            comm_metadata={
                "sharded": [],
                "shared": ["tp-sp1-sp2"],
                "reduce": ["sp1-sp2"],
            },
        )

        # modulate the skip with a conv
        if self.use_big_skip_conv:
            self.skip_conv = Conv2DLayer(
                in_chans,
                out_chans,
                kernel_size=1,
                bias=False,
                comm_metadata={
                    "sharded": [],
                    "shared": ["tp-sp1-sp2"],
                    "reduce": ["sp1-sp2"],
                },
            )

        self.apply(self._init_weights)

    def _init_coords(self):
        """coordinates for positional embedding"""
        t_coords = torch.linspace(0, 1, self.temporal_context_window_size)
        assert (
            self.domain_metadata is not None
        ), "data loader must provide domain metadata"
        # merge spatial and temporal coords
        H, W, coord_dim = self.domain_metadata["coords"].shape
        spatial_coords = self.domain_metadata["coords"].unsqueeze(0)  # (1, H, W, 3)
        t_coords = t_coords.view(
            self.temporal_context_window_size, 1, 1, 1
        )  # (T, 1, 1, 1)
        t_broadcast = t_coords.expand(-1, H, W, 1)  # (T, H, W, 1)
        spatial_broadcast = spatial_coords.expand(
            self.temporal_context_window_size, -1, -1, -1
        )  # (T, H, W, 3)
        coords = torch.cat([spatial_broadcast, t_broadcast], dim=-1)  # (T, H, W, 4)
        self.register_buffer("coords", coords, persistent=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif te is not None and isinstance(m, te.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or (
            te is not None and isinstance(m, te.LayerNorm)
        ):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.RMSNorm) or (
            te is not None and isinstance(m, te.RMSNorm)
        ):
            nn.init.constant_(m.weight, 1.0)

    @profile("embed")
    def prepare_tokens(self, x):
        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.patch_embed(x)  # patch linear embedding
        x = x.view(b, t, self.embed_dim, self.patch_embed.h, self.patch_embed.w)
        x = torch.einsum("btchw->bthwc", x)

        if self.coord_pos_embed:
            T, H, W, C = self.coords.shape
            pos = torch.einsum("thwc->tchw", self.coords)
            pos = self.pos_embed(pos)
            pos = torch.einsum("tchw->thwc", pos)
            return x + pos
        else:
            return x + self.pos_embed

    @profile("head")
    def forward_head(self, x):
        b, t, h, w, c = x.shape
        x = x.contiguous()
        # apply head
        x = self.head(x)
        x = x.reshape(
            shape=(b, t, h, w, self.patch_size, self.patch_size, self.out_chans)
        )
        x = torch.einsum("nthwpqc->ntchpwq", x)
        x = x.reshape(
            shape=(b, t, self.out_chans, h * self.patch_size, w * self.patch_size)
        )
        return x

    @profile("forward")
    def forward(self, x):
        b, t, c, h, w = x.shape

        if self.use_big_skip_conv:
            # for big skip residual
            skip = nn.Identity()(x)

        # transformer blocks
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.forward_head(x)

        if self.use_big_skip_conv:
            skip = skip.view(b * t, self.in_chans, h, w)
            skip = self.skip_conv(skip)
            skip = skip.view(b, t, self.out_chans, h, w)
            x = x + skip
            # if non-conv big skip (but might lead to artifacts in rollout)
            # x = x + skip[:, :, :self.out_chans]

        return x[:, -1:]


if __name__ == "__main__":
    batch_size = 2
    temporal_window = 4
    in_channels = 6
    out_channels = 6
    img_size = (720, 1440)
    patch_size = 8
    window_size = (9, 18)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain_metadata = create_dummy_metadata(img_size[0], img_size[1], in_channels)

    x = torch.randn(batch_size, temporal_window, in_channels, img_size[0], img_size[1], device=device)

    model = SwinTransformer(
        temporal_context_window_size=temporal_window,
        patch_size=patch_size,
        window_size=window_size,
        in_chans=in_channels,
        out_chans=out_channels,
        embed_dim=64,
        depth=2,
        num_heads=8,
        domain_metadata=domain_metadata,
    ).to(device)

    print(f"Input shape: {x.shape}")
    out = model(x)
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
