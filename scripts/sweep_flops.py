import os
import sys
import itertools
import hydra
from omegaconf import DictConfig
import numpy as np

# project root setup
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from utils.flops_utils import FlopsCalculator

patches = [4]
configs = [
    (192, 6),
    (256, 8),
    (384, 8),
    (512, 8),
    (512, 12),
    (640, 12),
    (768, 12),
    (768, 16),
    (1024, 12),
    (1024, 16),
    (1024, 24),
    (1536, 16),
]

global_batch_size = 16

budgets = [
    6E17,
    1E18,
    3E18,
    6E18,
    1E19,
    3E19,
    6E19,
]

@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    c_in = 71 + len(cfg.data.invariants)
    c_out = 71
    h, w = 720, 1440

    print(f"{'embed':>6} {'depth':>6} {'patch':>5} {'per_step_flop':>10} {'param(m)':>10} {'num_iters':>10} {'used_tflop':>12}")
    print("-" * 100)

    results = []

    for budget in budgets:
        # for embed, depth, patch_size in itertools.product(embeds, depths, patches):
        for embed, depth in configs:
            patch_size = patches[0]
            cfg_local = cfg.copy()
            cfg_local.model.embed_dim = embed
            cfg_local.model.depth = depth
            cfg_local.model.patch_size = patch_size
            cfg_local.train.temporal_context_window = 1

            fl = FlopsCalculator(cfg_local, h=h, w=w, c_in=c_in, c_out=c_out)
            per_step_flops = fl.flops() * 1E12 * 3 * global_batch_size
            params = fl.param_count()
            num_iters = budget / per_step_flops
            if num_iters < 1000 or num_iters > 120000:
                continue
            used_flops = per_step_flops * num_iters
            results.append((embed, depth, patch_size, per_step_flops, params, num_iters, used_flops))

    # Sort by used compute
    results.sort(key=lambda x: x[-1])

    for embed, depth, patch_size, per_step_flops, params, num_iters, used_flops in results:
        print(f"{embed:6} {depth:6} {patch_size:5} {per_step_flops:12.2e} {params:10.1f} {num_iters:10.0f} {used_flops:12.2e}")

    print("-" * 100)
    print("-" * 100)

    # Sort by param count
    results.sort(key=lambda x: x[-3])

    print(f"{'embed':>6} {'depth':>6} {'patch':>5} {'per_step_flop':>10} {'param(m)':>10} {'num_iters':>10} {'used_tflop':>12}")
    print("-" * 100)

    for embed, depth, patch_size, per_step_flops, params, num_iters, used_flops in results:
        print(f"{embed:6} {depth:6} {patch_size:5} {per_step_flops:12.2e} {params:10.1f} {num_iters:10.1f} {used_flops:12.2e}")


if __name__ == "__main__":
    main()
