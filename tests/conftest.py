"""Pytest plugins for **tests/** only. Hooks apply to tests collected under this tree."""

import logging
import os


def _primary_pytest_cli_process() -> bool:
    """Full pytest UI on rank 0 / xdist gw0 only; other ranks get quiet flags."""
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker is not None and worker != "gw0":
        return False
    for key in (
        "SLURM_PROCID",
        "PMI_RANK",
        "OMPI_COMM_WORLD_RANK",
        "MV2_COMM_WORLD_RANK",
        "RANK",
    ):
        v = os.environ.get(key)
        if v is not None:
            try:
                return int(v) == 0
            except ValueError:
                return True
    return True


def pytest_load_initial_conftests(early_config, parser, args):
    if not _primary_pytest_cli_process():
        for flag in ("--no-header", "--no-summary", "-qq", "--disable-warnings"):
            if flag not in args:
                args.append(flag)


def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:.*torch\.cuda\.amp\.autocast.*:FutureWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:.*CUDA_DEVICE_MAX_CONNECTIONS.*:UserWarning",
    )


def pytest_sessionstart(session):
    for name in (
        "torch._dynamo",
        "torch._dynamo.convert_frame",
        "torch._dynamo.eval_frame",
        "torch._inductor",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

    try:
        import torch._logging

        if hasattr(torch._logging, "set_logs"):
            torch._logging.set_logs(dynamo=logging.ERROR)
    except Exception:
        pass
