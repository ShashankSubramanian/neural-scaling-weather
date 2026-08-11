"""Analytical FLOP / param helpers for the Swin stack.

Positional embedding: estimates assume *coordinate-style* pos embed
(:class:`~models.swin.Swin` with ``coord_pos_embed=True``) — a
``PatchEmbedLayer`` on 4-channel coord maps (``in_chans=4``). Learnable
tensor pos embed (add-only, no conv) is not modeled here.
"""

import math
import numpy as np

TFLOPS = 1e-12


def attention_flops(batch, seq, embed, heads):
    head_dim = embed // heads
    qkv = 2 * batch * seq * embed * embed * 3 + batch * seq * embed * 3
    logits = 2 * batch * heads * seq * head_dim * seq
    softmax = 5 * batch * heads * seq * seq
    attend = 2 * batch * heads * seq * seq * head_dim
    project = 2 * batch * seq * embed * embed + batch * seq * embed
    return (qkv + logits + softmax + attend + project) * TFLOPS


def mlp_flops(batchseq, embed, hidden):
    fc1 = 2 * batchseq * embed * hidden + batchseq * hidden
    act = 8 * batchseq * hidden
    fc2 = 2 * batchseq * hidden * embed + batchseq * embed
    return (fc1 + act + fc2) * TFLOPS


def patch_embed_flops(batch, h, w, c, patch_size, embed):
    seq = h // patch_size * w // patch_size
    patchify = 2 * batch * h * w * c * embed + batch * seq * embed
    return patchify * TFLOPS


def layer_norm_flops(size):
    return 8 * size * TFLOPS


def rms_norm_flops(size):
    return 3 * size * TFLOPS


def pos_embed_flops(size):
    return size * TFLOPS


def head_flops(batch, h, w, embed, patch_size, out_chans):
    seq = h // patch_size * w // patch_size
    head = 2 * batch * seq * embed * out_chans * patch_size * patch_size
    return head * TFLOPS


class FlopsCalculator:
    def __init__(
        self,
        cfg,
        h,
        w,
        c_in,
        c_out,
    ):
        self.tera_factor = 1e-12
        self.cfg = cfg
        self.wh = self.cfg.model.window_size[0]
        self.ww = self.cfg.model.window_size[1]
        self.embed = self.cfg.model.embed_dim
        self.heads = self.cfg.model.num_heads
        self.depth = self.cfg.model.depth
        self.patch_size = self.cfg.model.patch_size
        self.batch = self.cfg.parallelism.micro_batch_size
        self.t = self.cfg.train.temporal_context_window
        self.seq = self.t * self.wh * self.ww
        self.h = h
        self.w = w
        self.h_patch = h // self.patch_size
        self.w_patch = w // self.patch_size
        self.num_windows = self.h_patch // self.wh * self.w_patch // self.ww
        self.c_in = c_in
        self.c_out = c_out
        self.params_backbone = 0

    def flops(self, verbose=False):
        patch = patch_embed_flops(
            self.batch * self.t, self.h, self.w, self.c_in, self.patch_size, self.embed
        )
        pos = patch_embed_flops(
            self.batch * self.t, self.h, self.w, 4, self.patch_size, self.embed
        )
        attn = attention_flops(
            self.batch * self.num_windows, self.seq, self.embed, self.heads
        )
        mlp = mlp_flops(
            self.batch * self.t * self.h_patch * self.w_patch,
            self.embed,
            4 * self.embed,
        )
        norm = rms_norm_flops(
            self.batch * self.t * self.h_patch * self.w_patch * self.embed
        )
        block_flops = attn + mlp + 2 * norm
        block_flops += 2 * norm # for qk norm
        head = head_flops(
            self.batch, self.h, self.w, self.embed, self.patch_size, self.c_out
        )
        # if self.c_in != self.c_out:
        #     skip_conv = (
        #         self.batch * self.t * self.h * self.w * self.c_out * self.c_in * TFLOPS
        #     )
        # else:
        #     skip_conv = 0
        # skip = skip_conv + self.batch * self.t * self.h * self.w * self.c_in * TFLOPS

        if verbose:
            print(f"flops in patch embedding: {patch} TFlops")
            print(f"flops in position embedding: {pos} TFlops")
            print(f"flops in attention: {attn} TFlops")
            print(f"flops in mlp: {mlp} TFlops")
            print(f"flops in norm: {norm} TFlops")
            print(f"flops in block: {block_flops} TFlops")
            print(f"flops in head: {head} TFlops")
            # print(f"flops in skip: {skip} TFlops")
        skip = 0

        total_flops = patch + pos + block_flops * self.depth + head + skip
        return total_flops

    def param_count(self):
        patch = self.embed * self.c_in * self.patch_size * self.patch_size + self.embed
        pos = self.embed * 4 * self.patch_size * self.patch_size + self.embed
        mlp = self.embed * 4 * self.embed * 2
        norm = self.embed * 1
        qkv = self.embed * self.embed * 3
        proj = self.embed * self.embed
        block = qkv + proj + mlp + norm * 4
        head = self.c_out * self.embed * self.patch_size * self.patch_size
        # if self.c_in != self.c_out:
        #     skip_conv = self.c_in * self.c_out
        # else:
        #     skip_conv = 0
        skip_conv = 0
        total_params = patch + pos + block * self.depth + head + skip_conv
        self.params_backbone = (block * self.depth) * 1e-6
        return total_params * 1e-6

