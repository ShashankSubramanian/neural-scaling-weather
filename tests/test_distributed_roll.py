import torch
import torch.distributed as dist
import unittest
from utils import comm

from test_helpers import setup_test_class

from parameterized import parameterized

from distributed.helpers import compute_split_shapes
from distributed.mappings import (
    split_forward_allgather_backward,
    allgather_forward_split_backward,
    roll_forward_reverseroll_backward,
)


class TestDistributedRoll(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_test_class(cls)

    @classmethod
    def tearDownClass(cls):
        if cls.world_size > 1:
            dist.destroy_process_group(None)

    @parameterized.expand([[1, 4, 4, 1, 1e-4], [4, 720, 1440, 5, 1e-4]])
    def test_distributed_block(self, batch, H, W, embed, tolerance):
        window_size = (2, 2)  # use window size that divides H,W evenly
        sh = -window_size[0] // 2
        sw = -window_size[1] // 2

        #############################################################
        # non-distributed op
        #############################################################
        # temporarily remove tp and tp-cp groups for non-distributed version
        old_groups = {}
        for gr_name in ["tp", "tp-sp1-sp2"]:
            old_groups[gr_name] = comm._COMM_GROUPS.pop(gr_name, None)

        # inp = torch.randn((batch, H, W, embed), dtype=torch.float32, device=self.device)
        inp = (
            torch.Tensor(torch.arange(batch * H * W * embed))
            .reshape((batch, H, W, embed))
            .float()
            .to(self.device)
        )
        h_dim, w_dim = 1, 2
        inp.requires_grad = True

        # forward pass
        out = torch.roll(inp, shifts=(sh, sw), dims=(h_dim, w_dim))

        # backward pass
        with torch.no_grad():
            out_grad = torch.randn_like(out)
        out.backward(out_grad)
        inp_grad = inp.grad.clone()

        #############################################################
        # distributed op
        #############################################################
        # restore groups for distributed version
        for gr_name, group in old_groups.items():
            if group is not None:
                comm._COMM_GROUPS[gr_name] = group

        # create distributed block with sp_shapes based on W only
        sp1_shapes = compute_split_shapes(H, comm.get_size("sp1"), patch_size=1)
        sp2_shapes = compute_split_shapes(W, comm.get_size("sp2"), patch_size=1)

        # split the input tensor
        with torch.no_grad():
            inp_split_2 = split_forward_allgather_backward(
                inp, dim=w_dim, comm_name="sp2", patch_size=1
            )
            inp_local = split_forward_allgather_backward(
                inp_split_2, dim=h_dim, comm_name="sp1", patch_size=1
            )
        inp_local.requires_grad = True

        # forward pass
        out_local = roll_forward_reverseroll_backward(
            inp_local, shifts=(sw, sh), dims=(w_dim, h_dim), comm_names=["sp2", "sp1"]
        )

        # backward pass local
        with torch.no_grad():
            out_grad_sp2 = split_forward_allgather_backward(
                out_grad, dim=w_dim, comm_name="sp2", patch_size=1
            )
            out_grad_local = split_forward_allgather_backward(
                out_grad_sp2, dim=h_dim, comm_name="sp1", patch_size=1
            )

        out_local.backward(out_grad_local)
        inp_grad_local = inp_local.grad.clone()

        #############################################################
        # evaluate forward pass
        #############################################################
        with torch.no_grad():
            out_gather_sp2 = allgather_forward_split_backward(
                out_local, dim=w_dim, shapes=sp2_shapes, comm_name="sp2"
            )
            out_gather = allgather_forward_split_backward(
                out_gather_sp2, dim=h_dim, shapes=sp1_shapes, comm_name="sp1"
            )
            # compute error over spatial dimensions
            err = torch.mean(
                torch.norm(out - out_gather, p=2, dim=(1, 2, 3))
                / torch.norm(out, p=2, dim=(1, 2, 3))
            )
            print(f"{self.world_rank} has error {err}")
            if self.print_to_screen:
                print(f"final relative error of output in block: {err.item()}")
        self.assertTrue(err.item() <= tolerance)

        #############################################################
        # evaluate backward pass
        #############################################################
        with torch.no_grad():
            inp_grad_gather_2 = allgather_forward_split_backward(
                inp_grad_local, dim=w_dim, shapes=sp2_shapes, comm_name="sp2"
            )
            inp_grad_gather = allgather_forward_split_backward(
                inp_grad_gather_2, dim=h_dim, shapes=sp1_shapes, comm_name="sp1"
            )
            err = torch.mean(
                torch.norm(inp_grad - inp_grad_gather, p=2, dim=(1, 2, 3))
                / torch.norm(inp_grad, p=2, dim=(1, 2, 3))
            )
            if self.print_to_screen:
                print(f"final relative error of gradients in block: {err.item()}")
        self.assertTrue(err.item() <= tolerance)


if __name__ == "__main__":
    unittest.main()
