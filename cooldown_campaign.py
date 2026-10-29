"""Submit the 57 base cooldowns for the p4 scaling campaign.

The default is the full 1979–2016 record under
pe-scaling-adamw-clip1-qknorm. Pass the same --fraction used for
pretrain_campaign.py. Cooldown iteration targets are not scaled, so a
shorter year window repeats data more often at the same FLOP budgets.
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

from pretrain_campaign import (
    BASE_NODES,
    CAMPAIGNS,
    CHECKPOINT_STEP,
    P4_CAMPAIGN,
    TRAIN_END_YEAR,
    approx_steps_per_epoch,
    build_sbatch_command,
    build_train_params,
)


HOST_REGISTRY_ROOT = Path("/pscratch/sd/s/shas1693/results/neuralscaling")
CONTAINER_REGISTRY_ROOT = Path("/registry")
COOLDOWN_FRACTION = 0.05

# One ladder per P4_CAMPAIGN entry, in the same order.
COOLDOWN_TO_ITERS = (
    (12_035, 20_059),
    (8_868, 14_780),
    (5_522, 9_203),
    (2_752, 4_587, 13_760, 27_520),
    (1_269, 2_115, 6_345, 12_691, 21_151),
    (2_992, 5_984, 9_973, 29_919, 59_838, 99_730),
    (1_736, 3_472, 5_786, 17_358, 34_716, 57_860),
    (2_349, 3_915, 11_744, 23_488, 39_147, 117_440),
    (7_640, 15_279, 25_465, 76_395),
    (5_364, 10_728, 17_880, 53_639, 107_278),
    (8_087, 13_478, 40_435, 80_870),
    (6_119, 10_198, 30_595, 61_190),
    (7_678, 23_035, 46_071),
    (15_417, 30_834),
    (10_371, 20_741),
)

if len(COOLDOWN_TO_ITERS) != len(P4_CAMPAIGN):
    raise ValueError("COOLDOWN_TO_ITERS and P4_CAMPAIGN must have equal lengths")


def cooldown_from_checkpoint_iter(cooldown_to_iter: int) -> int:
    """Return the latest saved checkpoint strictly below 95% of the target."""
    limit = (1.0 - COOLDOWN_FRACTION) * cooldown_to_iter
    checkpoint_index = math.ceil(limit / CHECKPOINT_STEP) - 1
    return 10 if checkpoint_index < 1 else checkpoint_index * CHECKPOINT_STEP


def build_plan(fraction: str) -> list[dict]:
    spec = CAMPAIGNS[fraction]
    run_name = spec["run_name"]
    planned = []
    for (
        (base_tag, (sp1, sp2), last_required_checkpoint),
        target_iterations,
    ) in zip(P4_CAMPAIGN, COOLDOWN_TO_ITERS):
        for to_iter in target_iterations:
            from_iter = cooldown_from_checkpoint_iter(to_iter)
            if from_iter > last_required_checkpoint:
                raise ValueError(
                    f"{base_tag} needs checkpoint {from_iter}, but its pretrain stops "
                    f"at {last_required_checkpoint}"
                )

            pretrain_tag = base_tag
            branch_from = (
                CONTAINER_REGISTRY_ROOT
                / run_name
                / pretrain_tag
                / "checkpoints"
                / f"ckpt_iter{from_iter}.tar"
            )
            host_checkpoint = (
                HOST_REGISTRY_ROOT
                / run_name
                / pretrain_tag
                / "checkpoints"
                / f"ckpt_iter{from_iter}.tar"
            )
            cooldown_tag = f"{base_tag}-cool{to_iter}-base"

            params = build_train_params(
                base_tag,
                sp1,
                sp2,
                max_iterations=to_iter,
                log_every=CHECKPOINT_STEP,
                train_start_year=spec["start_year"],
            )
            params.update(
                {
                    "train.branch_from": str(branch_from),
                    "optimizer.scheduler": "cooldown",
                    "optimizer.cooldown_from_iter": from_iter,
                    "optimizer.cooldown_to_iter": to_iter,
                    "optimizer.cooldown_fraction": COOLDOWN_FRACTION,
                }
            )
            planned.append(
                {
                    "run_tag": cooldown_tag,
                    "base_tag": base_tag,
                    "pretrain_tag": pretrain_tag,
                    "from_iter": from_iter,
                    "to_iter": to_iter,
                    "sp1": sp1,
                    "sp2": sp2,
                    "host_checkpoint": host_checkpoint,
                    "command": build_sbatch_command(
                        cooldown_tag, params, sp1, sp2, run_name=run_name
                    ),
                }
            )

    if len(planned) != 57:
        raise ValueError(f"Expected 57 cooldown jobs, built {len(planned)}")
    return planned


def missing_checkpoints(planned: list[dict]) -> list[Path]:
    required = {job["host_checkpoint"] for job in planned}
    return sorted(path for path in required if not path.is_file())


def submit_plan(
    planned: list[dict],
    fraction: str,
    assume_yes: bool,
    skip_checkpoint_checks: bool,
) -> None:
    if not skip_checkpoint_checks:
        missing = missing_checkpoints(planned)
        if missing:
            print(
                f"Refusing to submit: {len(missing)} required pretrain checkpoint(s) "
                "are missing:",
                file=sys.stderr,
            )
            for path in missing:
                print(f" - {path}", file=sys.stderr)
            print(
                "Wait for or resubmit pretrain_campaign.py. Use "
                "--skip-checkpoint-checks only if the registry mount changed.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    spec = CAMPAIGNS[fraction]
    steps_per_epoch = approx_steps_per_epoch(spec["start_year"])
    full_steps = approx_steps_per_epoch(1979)
    print(
        f"Cooldown campaign: {len(planned)} jobs ({fraction} data) "
        f"for run_name={spec['run_name']}"
    )
    print(
        f"training years={spec['start_year']}-{TRAIN_END_YEAR}; "
        f"cooldown fraction={COOLDOWN_FRACTION}; "
        f"~{steps_per_epoch:.0f} steps/epoch "
        f"(full record ~{full_steps:.0f})"
    )
    for index, job in enumerate(planned, start=1):
        nodes = BASE_NODES * job["sp1"] * job["sp2"]
        epochs = job["to_iter"] / steps_per_epoch
        print(
            f"\n[{index:02d}/{len(planned)}] {job['run_tag']} "
            f"branch={job['pretrain_tag']}:{job['from_iter']} "
            f"target={job['to_iter']} (~{epochs:.1f} epochs) "
            f"sp=({job['sp1']},{job['sp2']}) nodes={nodes}"
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
            "full is 1979-2016; must match the pretrain_campaign.py fraction. "
            "Iteration targets stay fixed"
        ),
    )
    parser.add_argument(
        "--yes", action="store_true", help="submit without the confirmation prompt"
    )
    parser.add_argument(
        "--skip-checkpoint-checks",
        action="store_true",
        help="do not verify source checkpoints under the host registry path",
    )
    args = parser.parse_args()
    submit_plan(
        build_plan(args.fraction),
        args.fraction,
        assume_yes=args.yes,
        skip_checkpoint_checks=args.skip_checkpoint_checks,
    )


if __name__ == "__main__":
    main()
