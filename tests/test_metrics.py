import os
import numpy as np
import torch
import torch.distributed as dist
import unittest
from utils import comm
from utils.losses import WeightedRMSE
from parameterized import parameterized
from test_helpers import (
    setup_test_class,
)
from utils.misc_utils import (
    create_dummy_metadata,
    split_dummy_metadata,
)

from distributed.helpers import (
    compute_split_shapes,
)
from distributed.mappings import (
    split_forward_allgather_backward,
    allgather_forward_split_backward,
)


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_test_class(cls, True, "test_metrics")

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group(None)

    @parameterized.expand(
        [
            [4, 2, 32, 720, 1440, 1e-4],
        ]
    )
    def test_weighted_rmse(
        self,
        B,
        T,
        C,
        H,
        W,
        tolerance,
    ):

        #############################################################
        # non-distributed op
        #############################################################
        old_groups = {}
        comm_groups = list(comm._COMM_GROUPS.keys())
        for old_group in comm_groups:
            old_groups[old_group] = comm._COMM_GROUPS.pop(old_group, None)

        # create a random tensor
        pred = torch.randn(B, T, C, H, W, dtype=torch.float32, device=self.device)
        target = torch.randn(B, T, C, H, W, dtype=torch.float32, device=self.device)

        # create a weighted rmse loss
        metadata = create_dummy_metadata(H, W, C)
        metric = WeightedRMSE(weights=metadata["area_weights"]).to(self.device)
        rmse = metric(pred, target)

        #############################################################
        # distributed op
        #############################################################
        # restore groups for distributed version
        for gr_name in old_groups:
            comm._COMM_GROUPS[gr_name] = old_groups[gr_name]

        sp1_shapes = compute_split_shapes(H, comm.get_size("sp1"))
        sp2_shapes = compute_split_shapes(W, comm.get_size("sp2"))
        with torch.no_grad():
            pred = split_forward_allgather_backward(
                pred, dim=0, comm_name="dp", shapes=None
            )
            pred = split_forward_allgather_backward(
                pred, dim=-2, comm_name="sp1", shapes=sp1_shapes
            )
            pred = split_forward_allgather_backward(
                pred, dim=-1, comm_name="sp2", shapes=sp2_shapes
            )
            target = split_forward_allgather_backward(
                target, dim=0, comm_name="dp", shapes=None
            )
            target = split_forward_allgather_backward(
                target, dim=-2, comm_name="sp1", shapes=sp1_shapes
            )
            target = split_forward_allgather_backward(
                target, dim=-1, comm_name="sp2", shapes=sp2_shapes
            )

        metadata = split_dummy_metadata(metadata, sp1_shapes, sp2_shapes)
        metric = WeightedRMSE(weights=metadata["area_weights"]).to(self.device)
        rmse_distributed = metric(pred, target)
        dist.all_reduce(
            rmse_distributed,
            op=torch.distributed.ReduceOp.AVG,
            group=comm.get_group("dp"),
        )

        err = (rmse - rmse_distributed).norm().item()

        if self.print_to_screen:
            print(f"final relative error of rmse: {err}")

        self.assertTrue(err <= tolerance)


if __name__ == "__main__":
    unittest.main()
