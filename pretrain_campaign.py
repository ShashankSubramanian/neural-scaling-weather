"""Submit the base pretrains for the p4 scaling campaign.

The default is the full 1979–2016 record under
pe-scaling-adamw-clip1-qknorm. Every fraction uses the same pretrain stops:
the last checkpoint its cooldown ladder needs.
"""

import argparse
import math
import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SUBMIT_SCRIPT = REPO_ROOT / "submit_batch.sh"

PARTITION = "gpu&hbm40g"
BASE_NODES = 4
BATCH_SIZE = 16
TRAIN_END_YEAR = 2016
# Hourly ERA5 starts. Printed epoch counts use this; training itself does not.
HOURS_PER_YEAR = 365.25 * 24
ROLLOUT_STEPS = 1
HEAD_DIM = 64
CHECKPOINT_STEP = 2000

# Run names stay unique so a resubmit cannot resume a different fraction's
# checkpoints. Pretrain length does not depend on the fraction.
CAMPAIGNS = {
    "full": {
        "start_year": 1979,
        "run_name": "pe-scaling-adamw-clip1-qknorm",
    },
    "half": {
        "start_year": 1998,
        "run_name": "pe-scaling-half-1998-te-adamw-clip1-qknorm",
    },
    "eighth": {
        "start_year": 2012,  # 2012–2016: 5 years, the nearest whole-year eighth
        "run_name": "pe-scaling-eighth-2012-te-adamw-clip1-qknorm",
    },
}

RUN_NAME = CAMPAIGNS["full"]["run_name"]
TRAIN_START_YEAR = CAMPAIGNS["full"]["start_year"]

# (base tag, (sp1, sp2), last checkpoint needed by its cooldown ladder).
P4_CAMPAIGN = (
    ("p4-e96-d4-lr1p5em3", (1, 1), 18_000),
    ("p4-e96-d6-lr1p5em3", (1, 1), 14_000),
    ("p4-e128-d6-lr1p5em3", (1, 1), 8_000),
    ("p4-e192-d6-lr1p5em3", (1, 1), 26_000),
    ("p4-e256-d8-lr1p5em3", (1, 1), 20_000),
    ("p4-e384-d8-lr1p5em3", (1, 1), 94_000),
    ("p4-e512-d8-lr1p2em3", (1, 1), 54_000),
    ("p4-e512-d12-lr1em3", (1, 1), 110_000),
    ("p4-e640-d12-lr8em4", (1, 1), 72_000),
    ("p4-e768-d12-lr6em4", (1, 1), 100_000),
    ("p4-e768-d16-lr5em4", (1, 1), 76_000),
    ("p4-e1024-d12-lr5em4", (1, 1), 58_000),
    ("p4-e1024-d16-lr5em4", (1, 2), 42_000),
    ("p4-e1024-d24-lr5em4", (1, 2), 28_000),
    ("p4-e1536-d16-lr5em4", (2, 2), 18_000),
)


def get_window_size(
    new_p: int,
    sp1: int,
    sp2: int,
    img_size=(720, 1440),
    base_p: int = 4,
    base_window_size=(9, 18),
) -> tuple[int, int]:
    """Choose a physical-scale-matched window compatible with local domains."""
    height, width = img_size
    base_window_height, _ = base_window_size
    scaled_height = max(1, math.ceil(base_window_height * base_p / new_p))
    height_grid, width_grid = height // new_p, width // new_p
    possible_heights = [
        value for value in range(1, height_grid + 1) if height_grid % value == 0
    ]
    valid_heights = [
        value
        for value in possible_heights
        if 2 * value <= width_grid and width_grid % (2 * value) == 0
    ]
    window_height = max(
        (value for value in valid_heights if value <= scaled_height), default=1
    )
    window_width = 2 * window_height

    assert height_grid % window_height == 0
    assert width_grid % window_width == 0
    assert ((height // sp1) // new_p) % window_height == 0
    assert ((width // sp2) // new_p) % window_width == 0
    return window_height, window_width


def parse_scaling_tag(tag: str) -> dict:
    """Parse tags such as p4-e192-d6-lr1p5em3 (optionally ending in -saveN)."""
    parse_tag = re.sub(r"-save\d+$", "", tag)
    match = re.fullmatch(
        r"p(\d+)-e(\d+)-d(\d+)-lr(?:(\d+)p(\d+)|(\d+))em(\d+)",
        parse_tag,
    )
    if match is None:
        raise ValueError(f"Unrecognized scaling tag: {tag!r}")

    patch_size, embed_dim, depth = (
        int(match.group(index)) for index in (1, 2, 3)
    )
    mantissa = (
        f"{match.group(4)}.{match.group(5)}"
        if match.group(4) is not None
        else match.group(6)
    )
    lr = float(Decimal(mantissa) * Decimal(10) ** -int(match.group(7)))
    return {
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "depth": depth,
        "lr": lr,
    }


def approx_steps_per_epoch(start_year: int) -> float:
    """Approximate optimizer steps to pass once over hourly samples."""
    n_years = TRAIN_END_YEAR - start_year + 1
    return n_years * HOURS_PER_YEAR / BATCH_SIZE


def build_train_params(
    base_tag: str,
    sp1: int,
    sp2: int,
    max_iterations: int,
    log_every: int = CHECKPOINT_STEP,
    train_start_year: int = TRAIN_START_YEAR,
) -> dict:
    config = parse_scaling_tag(base_tag)
    patch_size = config["patch_size"]
    embed_dim = config["embed_dim"]
    window_size = get_window_size(patch_size, sp1, sp2)
    return {
        "data.loader": "dali",
        "data.train_start_year": train_start_year,
        "data.batch_size": BATCH_SIZE,
        "optimizer": "adamw",
        "optimizer.lr": config["lr"],
        "optimizer.max_iterations": max_iterations,
        "parallelism.use_transformer_engine": True,
        "parallelism.micro_batch_size": 1,
        "parallelism.sp1": sp1,
        "parallelism.sp2": sp2,
        "train.num_rollout_steps": ROLLOUT_STEPS,
        "train.clip_grad_norm": 1.0,
        "train.log_every": log_every,
        "model.patch_size": patch_size,
        "model.embed_dim": embed_dim,
        "model.depth": config["depth"],
        "model.num_heads": embed_dim // HEAD_DIM,
        "model.window_size": f"[{window_size[0]},{window_size[1]}]",
    }


def kv_to_arg_strings(params: dict) -> list[str]:
    def format_value(value):
        return str(value).lower() if isinstance(value, bool) else value

    return [f"{key}={format_value(value)}" for key, value in params.items()]


def build_sbatch_command(
    run_tag: str,
    params: dict,
    sp1: int,
    sp2: int,
    run_name: str = RUN_NAME,
) -> list[str]:
    nodes = BASE_NODES * sp1 * sp2
    return [
        "sbatch",
        f"--job-name={run_name}",
        f"--nodes={nodes}",
        f"--chdir={REPO_ROOT}",
        "-C",
        PARTITION,
        str(SUBMIT_SCRIPT),
        f"run_name={run_name}",
        f"run_tag={run_tag}",
        *kv_to_arg_strings(params),
    ]


def build_plan(fraction: str) -> list[dict]:
    spec = CAMPAIGNS[fraction]
    planned = []
    for base_tag, (sp1, sp2), max_iterations in P4_CAMPAIGN:
        params = build_train_params(
            base_tag,
            sp1,
            sp2,
            max_iterations,
            train_start_year=spec["start_year"],
        )
        planned.append(
            {
                "run_tag": base_tag,
                "sp1": sp1,
                "sp2": sp2,
                "max_iterations": max_iterations,
                "params": params,
                "command": build_sbatch_command(
                    base_tag, params, sp1, sp2, run_name=spec["run_name"]
                ),
            }
        )
    return planned


def submit_plan(planned: list[dict], fraction: str, assume_yes: bool) -> None:
    spec = CAMPAIGNS[fraction]
    steps_per_epoch = approx_steps_per_epoch(spec["start_year"])
    longest = max(job["max_iterations"] for job in planned)
    full_steps = approx_steps_per_epoch(1979)
    print(f"Pretrain campaign: {len(planned)} base jobs ({fraction} data)")
    print(
        f"run_name={spec['run_name']}; "
        f"training years={spec['start_year']}-{TRAIN_END_YEAR}"
    )
    print(
        f"~{steps_per_epoch:.0f} steps/epoch; longest pretrain is "
        f"~{longest / steps_per_epoch:.0f} epochs "
        f"(full record would be ~{longest / full_steps:.0f} at the same step)"
    )
    for index, job in enumerate(planned, start=1):
        nodes = BASE_NODES * job["sp1"] * job["sp2"]
        epochs = job["max_iterations"] / steps_per_epoch
        print(
            f"\n[{index:02d}/{len(planned)}] {job['run_tag']} "
            f"sp=({job['sp1']},{job['sp2']}) nodes={nodes} "
            f"max_iterations={job['max_iterations']} (~{epochs:.1f} epochs)"
        )
        print(" ".join(job["command"]))

    if not assume_yes:
        try:
            response = input(f"\nSubmit all {len(planned)} jobs? [y/N]: ").lower()
        except EOFError:
            response = ""
        if response not in ("y", "yes"):
            print("Cancelled; no jobs submitted.")
            return

    failures = []
    for job in planned:
        try:
            subprocess.run(job["command"], check=True)
            print(f"Submitted {CAMPAIGNS[fraction]['run_name']}/{job['run_tag']}")
        except subprocess.CalledProcessError as error:
            failures.append((job["run_tag"], error.returncode))
            print(
                f"Submission failed for {job['run_tag']}: {error.returncode}",
                file=sys.stderr,
            )

    if failures:
        print(f"{len(failures)} submission(s) failed: {failures}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fraction",
        choices=tuple(CAMPAIGNS),
        default="full",
        help=(
            "full is 1979-2016; half and eighth are later slices. "
            "Pretrain stops match the cooldown checkpoints for every fraction"
        ),
    )
    parser.add_argument(
        "--yes", action="store_true", help="submit without the confirmation prompt"
    )
    args = parser.parse_args()
    submit_plan(build_plan(args.fraction), args.fraction, assume_yes=args.yes)


if __name__ == "__main__":
    main()
