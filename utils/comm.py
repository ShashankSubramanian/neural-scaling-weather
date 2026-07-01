import os
import logging
from utils.rank_generator import RankGenerator
import torch
import torch.distributed as dist

# dummy placeholder
_COMM_GROUPS = {}

# routines for specific comm groups
def get_names():
    """Returns the names of all available communicators."""
    return _COMM_GROUPS.keys()


def is_initialized(comm_name):
    """check if initialized."""
    return comm_name in _COMM_GROUPS


def get_group(comm_name):
    """Returns the group of a specified communicator."""
    if not is_initialized(comm_name):
        raise IndexError(f"Error, comm {comm_name} not initialized.")
    return _COMM_GROUPS[comm_name]


def get_size(comm_name):
    """Returns the size of a specified communicator."""
    if (not dist.is_initialized()) or (not is_initialized(comm_name)):
        return 1
    else:
        return dist.get_world_size(group=get_group(comm_name))


def get_rank(comm_name):
    """Returns the rank in a specified communicator."""
    if (not dist.is_initialized()) or (not is_initialized(comm_name)):
        return 0
    else:
        return dist.get_rank(group=get_group(comm_name))


def get_root(comm_name):
    """Returns the minimum rank (root) of a specified communicator."""
    if (not dist.is_initialized()) or (not is_initialized(comm_name)):
        return 0
    else:
        group = get_group(comm_name)
        ranks = dist.get_process_group_ranks(group)
        return min(ranks)

# routines for world comms
def get_world_size():
    """Returns the world size"""
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()


def get_world_rank():
    """Returns the world rank"""
    if not dist.is_initialized():
        return 0
    else:
        return dist.get_rank()


def get_local_rank():
    """Returns the local rank of the current process."""
    if not dist.is_initialized():
        return 0
    else:
        if os.getenv("LOCAL_RANK") is not None:
            # Use env var if available
            return int(os.getenv("LOCAL_RANK"))
        else:
            return get_world_rank() % torch.cuda.device_count()


def init(cfg):
    # init torch.distributed
    backend = cfg.parallelism.backend
    init_process_group(backend)

    # set model parallel sizes
    tp = cfg.parallelism.tp
    sp1 = cfg.parallelism.sp1
    sp2 = cfg.parallelism.sp2
    pp = cfg.parallelism.pp
    assert pp == 1, "ERROR: pipeline parallel not implemented"
    model_parallel_size = tp * sp1 * sp2 * pp
    world_size = get_world_size()
    assert (
        world_size % model_parallel_size == 0
    ), "ERROR: world size {} must be divisible by model parallel size {} (TP={}, SP=({}, {}), PP={})".format(
        world_size, model_parallel_size, tp, sp1, sp2, pp
    )
    dp = world_size // model_parallel_size
    assert dp >= 1, "ERROR: data parallel wireup failed since dp = {}".format(dp)
    world_rank = get_world_rank()
    if world_rank == 0:
        logging.info(f"Using backend {backend}")
        logging.info("Setting DP = {}, TP = {}, SP = ({}, {}), PP = {}".format(dp, tp, sp1, sp2, pp))

    # init model + dp groups individually
    init_model_parallel_info(
        tp=tp,
        sp1=sp1,
        sp2=sp2,
        dp=dp,
        pp=pp,
        order=cfg.parallelism.order,
    )

def init_process_group(backend):
    """init torch distributed process group
    """
    if os.getenv("WORLD_SIZE") is not None and os.getenv("RANK") is not None:
        world_size = int(os.getenv("WORLD_SIZE", 1))
        world_rank = int(os.getenv("RANK", 0))
    else:
        world_size = int(os.getenv("SLURM_NTASKS", 1))
        world_rank = int(os.getenv("SLURM_PROCID", 0))
    dist.init_process_group(backend=backend, rank=world_rank, world_size=world_size)

def init_model_parallel_info(tp=1, pp=1, dp=1, sp1=1, sp2=1, order="sp1-sp2-tp-pp-dp"):
    world_rank = get_world_rank()

    # builds a hyper-rectangle of GPU groups
    rank_gen = RankGenerator(
        tp=tp,
        dp=dp,
        pp=pp,
        sp1=sp1,
        sp2=sp2,
        order=order,
    )

    # build the different parallel groups
    global _COMM_GROUPS  # others need access to this
    groups_to_build = ["dp", "tp", "sp1", "sp2", "sp1-sp2", "pp", "tp-sp1-sp2", "dp-sp1-sp2"]
    for grp in groups_to_build:
        for ranks in rank_gen.get_ranks(grp):
            group = dist.new_group(ranks)
            if world_rank in ranks:
                _COMM_GROUPS[grp] = group


def all_model_groups_exist(model):
    """check if all model parallel groups exist"""
    for param in model.parameters():
        if hasattr(param, "comm_metadata"):
            for comm_name in param.comm_metadata["shared"]:
                if comm_name is not None and not is_initialized(comm_name):
                    assert False, f"Comm group '{comm_name}' not initialized for parameter '{param.name}'"
            for comm_name in param.comm_metadata["reduce"]:
                if comm_name is not None and not is_initialized(comm_name):
                    assert False, f"Comm group '{comm_name}' not initialized for parameter '{param.name}'"
            for comm_name in param.comm_metadata["sharded"]:
                if comm_name is not None and not is_initialized(comm_name):
                    assert False, f"Comm group '{comm_name}' not initialized for parameter '{param.name}'"
    return model
