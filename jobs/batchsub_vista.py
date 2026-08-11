import math
import pprint
import subprocess
import sys


def get_window_size(
    new_p, sp1, sp2, img_size=[720, 1440], base_p=4, base_window_size=[9, 18]
):
    H, W = img_size
    base_wh, _ = base_window_size
    scaled_wh = max(1, math.ceil(base_wh * base_p / new_p))
    H_grid, W_grid = H // new_p, W // new_p
    possible_wh = [h for h in range(1, H_grid + 1) if H_grid % h == 0]
    valid_wh = [
        h for h in possible_wh if (2 * h) <= W_grid and W_grid % (2 * h) == 0
    ]
    wh = max([h for h in valid_wh if h <= scaled_wh], default=1)
    ww = wh * 2

    assert (H // new_p) % wh == 0, "global sizes and window sizes not divisible"
    assert (W // new_p) % ww == 0, "global sizes and window sizes not divisible"
    assert ((H // sp1) // new_p) % wh == 0, (
        "local sizes and window sizes not divisible"
    )
    assert ((W // sp2) // new_p) % ww == 0, (
        "local sizes and window sizes not divisible"
    )

    return wh, ww


##########################################################################################
# Configuration
##########################################################################################
mode = "train"  # or "inference"
include_cooldown = False
run_name = "scaling"
run_tag = "p4-e1024-d16-lr5em4"
scheduler_selector = ("-p", "gh")
nodes = 8
batch_size = 16
rollout_steps = 1
head_dim = 64
embed_dim = 1024
depth = 16
num_heads = embed_dim // head_dim
patch_size = 4
lr = 5e-4
sp1 = 1
sp2 = 2
window_size = get_window_size(
    patch_size, sp1, sp2, img_size=[720, 1440], base_p=4, base_window_size=[9, 18]
)
if include_cooldown:
    cooldown_to_iter = 50000
    cooldown_from_iter = 42000
    cooldown_fraction = 0.05
    cooldown_branch_from = (
        f"/registry/{run_name}/{run_tag}/checkpoints/ckpt_iter{cooldown_from_iter}.tar"
    )
    run_tag += f"-cool{cooldown_to_iter}"
##########################################################################################


train_params = {
    "data.batch_size": batch_size,
    "optimizer.lr": lr,
    "parallelism.sp2": sp2,
    "parallelism.sp1": sp1,
    "train.num_rollout_steps": rollout_steps,
    "model.patch_size": patch_size,
    "model.embed_dim": embed_dim,
    "model.depth": depth,
    "model.num_heads": num_heads,
    "model.window_size": f"[{window_size[0]},{window_size[1]}]",
    "optimizer": "adamw",
    "train.clip_grad_norm": 1.0,
    "optimizer.max_iterations": 200000,
    "parallelism.micro_batch_size": 1,
}

if include_cooldown:
    cooldown_params = {
        "train.branch_from": cooldown_branch_from,
        "optimizer.scheduler": "cooldown",
        "optimizer.cooldown_to_iter": cooldown_to_iter,
        "optimizer.cooldown_from_iter": cooldown_from_iter,
        "optimizer.cooldown_fraction": cooldown_fraction,
    }

inference_params = {
    "inference.checkpoint": f"/registry/{run_name}/{run_tag}/checkpoints/ckpt_best.tar",
    "inference.checkpoint_hyperparams": (
        f"/registry/{run_name}/{run_tag}/hyperparameters.yaml"
    ),
    "parallelism.sp2": 1,
    "parallelism.sp1": 1,
}


def kv_to_arg_strings(params: dict) -> list[str]:
    return [f"{k}={v}" for k, v in params.items()]


def build_args_array(mode_val: str) -> list[str]:
    if mode_val == "train":
        args = kv_to_arg_strings(train_params)
        if include_cooldown:
            args += kv_to_arg_strings(cooldown_params)
        return args
    return kv_to_arg_strings(inference_params)


def build_sbatch_command(
    run_tag: str, args_list: list[str], submit_script: str
) -> list[str]:
    return [
        "sbatch",
        f"--job-name={run_name}",
        f"--nodes={nodes}",
        *scheduler_selector,
        submit_script,
        f"run_name={run_name}",
        f"run_tag={run_tag}",
        *args_list,
    ]


args_list = build_args_array(mode)
print("-" * 100)
print(f"Submitting with args for {run_name}/{run_tag} in {mode} mode:")
pprint.pprint(args_list)
print("-" * 100)
submit_script = (
    "submit_batch_inference.sh" if mode == "inference" else "submit_batch.sh"
)

cmd = build_sbatch_command(run_tag, args_list, submit_script)
print("-" * 100)
print("Sbatch command:", " ".join(cmd))
print("-" * 100)

try:
    resp = input("Proceed to submit with sbatch? [y/N]: ").strip().lower()
except EOFError:
    resp = ""
if resp in ("y", "yes"):
    try:
        subprocess.run(cmd, check=True)
        print(f"Submitted {run_name}")
    except subprocess.CalledProcessError as e:
        print(f"Submission failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
else:
    print("Cancelled; not submitted.")
