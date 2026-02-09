import os
import torch
import torch.distributed as dist
import unittest
from utils import comm

from models.swin_helpers import window_partition, window_reverse

from distributed.helpers import (
    compute_split_shapes_for_patching,
)
from distributed.mappings import (
    split_forward_allgather_backward,
    allgather_forward_split_backward,
)

from parameterized import parameterized

from test_helpers import (
    setup_test_class,
)

# from ic_logger import setup_ic_logger


class TestWindows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_test_class(cls, True, "test_windows")
        # setup_ic_logger("test_all")

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group(None)

    @parameterized.expand(
        [
            [4, 8, 16, 32, 8, 1e-4],
        ]
    )
    def test_distributed_model(
        self,
        B,
        T,
        H,
        W,
        C,
        tolerance,
    ):
        #############################################################
        # non-distributed op
        #############################################################
        # temporarily remove tp and tp-sp groups for non-distributed version
        old_groups = {}
        comm_groups = list(comm._COMM_GROUPS.keys())
        for old_group in comm_groups:
            old_groups[old_group] = comm._COMM_GROUPS.pop(old_group, None)
        

        # create tensor in BHWC format
        inp = torch.randn(
            (B, T, H, W, C),
            dtype=torch.float32,
            device=self.device,
        )

        windows = window_partition(inp, (4, 8))
        if self.print_to_screen:
            print(f"windows shape: {windows.shape}")
        inp_back = window_reverse(windows, (4, 8), (H, W))
        if self.print_to_screen:
            print(f"inp_back shape: {inp_back.shape}")
        err = torch.norm(inp - inp_back, p=2) / torch.norm(inp, p=2)
        if self.print_to_screen:
            print(f"windows and back seq error: {err.item()}")
        self.assertTrue(err.item() <= tolerance)

        #############################################################
        # distributed op
        #############################################################
        # restore groups for distributed version
        for group_name, group in old_groups.items():
            if group is not None:
                comm._COMM_GROUPS[group_name] = group

        # compute split shapes : split the input image based on what the patch size will be
        sp1_shapes = compute_split_shapes_for_patching(H, comm.get_size("sp1"), 1)
        sp2_shapes = compute_split_shapes_for_patching(W, comm.get_size("sp2"), 1)
        H_local = sp1_shapes[comm.get_rank("sp1")]
        W_local = sp2_shapes[comm.get_rank("sp2")]

        with torch.no_grad():
            # dp split
            inp_local = split_forward_allgather_backward(
                inp, dim=0, comm_name="dp", shapes=None
            )
            inp_local = split_forward_allgather_backward(
                inp_local, dim=-3, comm_name="sp1", shapes=sp1_shapes
            )
            inp_local = split_forward_allgather_backward(
                inp_local, dim=-2, comm_name="sp2", shapes=sp2_shapes
            )
        windows_local = window_partition(inp_local, (4, 8))
        inp_local_back = window_reverse(windows_local, (4, 8), (H_local, W_local))
        if self.print_to_screen:
            print(f"windows_local shape: {windows_local.shape}")
            print(f"inp_local_back shape: {inp_local_back.shape}")

        err = torch.norm(inp_local - inp_local_back, p=2) / torch.norm(inp_local, p=2)
        print(f"windows and back error at rank {self.world_rank}: {err.item()}")
        self.assertTrue(err.item() <= tolerance)

        with torch.no_grad():
            # compute error over spatial dimensions
            inp_gather = allgather_forward_split_backward(
                inp_local_back, dim=0, shapes=None, comm_name="dp"
            )
            inp_gather = allgather_forward_split_backward(
                inp_gather, dim=-3, comm_name="sp1", shapes=sp1_shapes
            )
            inp_gather = allgather_forward_split_backward(
                inp_gather, dim=-2, comm_name="sp2", shapes=sp2_shapes
            )

        if self.print_to_screen:
            print(f"inp_gather shape: {inp_gather.shape}")
            print(f"inp shape: {inp.shape}")

        err = torch.norm(inp - inp_gather, p=2) / torch.norm(inp, p=2)
        if self.print_to_screen:
            print(f"rel error : {err.item()}")
        self.assertTrue(err.item() <= tolerance)

if __name__ == "__main__":
    unittest.main()
