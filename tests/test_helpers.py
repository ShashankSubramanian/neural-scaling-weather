from typing import Optional
import os
import torch
from utils import comm

# from ic_logger import setup_ic_logger


def setup_test_class(cls, ic_logger: bool=False, prefix: Optional[str]=None):
    backend = os.getenv("BACKEND", "nccl")
    use_cuda = backend == "nccl" and torch.cuda.is_available()

    comm.init_process_group(backend)

    # if ic_logger:
    #     setup_ic_logger(prefix)

    cls.backend = backend
    cls.world_size = comm.get_world_size()
    cls.world_rank = comm.get_world_rank()
    cls.local_rank = comm.get_local_rank()

    # get model parallel sizes
    tp = int(os.getenv("TP", 1))
    sp1 = int(os.getenv("SP1", 1))
    sp2 = int(os.getenv("SP2", 1))
    pp = 1
    order = "sp1-sp2-tp-dp-pp"
    model_parallel_size = tp * sp1 * sp2 * pp
    dp = cls.world_size // model_parallel_size
    assert dp >= 1, "ERROR: data parallel wireup failed since dp = {}".format(dp)

    cls.print_to_screen = cls.world_rank == 0
    if cls.print_to_screen:
        print(
            "Distributed unit tests with DP = {}, TP = {}, SP = {}, {}, PP = {}".format(
                dp, tp, sp1, sp2, pp
            )
        )
        print(f"{backend=}")

    if use_cuda:
        if cls.print_to_screen:
            print("Running test on GPU")
        cls.device = torch.device(f"cuda:{cls.local_rank}")
        torch.cuda.set_device(cls.local_rank)
        torch.cuda.manual_seed(333)
    else:
        if cls.print_to_screen:
            print("Running test on CPU")
        cls.device = torch.device("cpu")
    torch.manual_seed(333)

    # init model parallel grps with dp grps
    comm.init_model_parallel_info(tp=tp, sp1=sp1, sp2=sp2, dp=dp, pp=pp, order=order)