import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from utils import comm
from functools import partial
from utils.profiler_utils import profile, profile_range
import logging

# torch utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# helper functions
from distributed.helpers import (
    _reduce,
    _split,
    _gather,
    _reduce_scatter,
    compute_split_shapes,
    _distributed_roll,
    _transpose,
)


class _IdentityForwardAllReduceBackward(torch.autograd.Function):
    """Identity forward, AllReduce in backward"""

    @staticmethod
    def forward(ctx, input_, comm_name_):
        ctx.comm_name = comm_name_
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, comm_name=ctx.comm_name), None


class _AllReduceForwardIdentityBackward(torch.autograd.Function):
    """AllReduce in forward, Identity in backward"""

    @staticmethod
    def forward(ctx, input_, comm_name_):
        return _reduce(input_, comm_name=comm_name_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _AllGatherForwardSplitBackward(torch.autograd.Function):
    """AllGather in forward and Split in backward"""

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, comm_name_):
        ctx.dim = dim_
        ctx.comm_name = comm_name_
        ctx.shapes = shapes_
        return _gather(input_, dim_, shapes_, comm_name_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.dim, ctx.comm_name, ctx.shapes), None, None, None


class _SplitForwardAllGatherBackward(torch.autograd.Function):
    """Split in forward and AllGather in backward"""

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, comm_name_):
        ctx.dim = dim_
        ctx.comm_name = comm_name_
        ctx.shapes = shapes_
        return _split(input_, dim_, comm_name_, sections=ctx.shapes)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _gather(grad_output, ctx.dim, ctx.shapes, ctx.comm_name),
            None,
            None,
            None
        )


class _ReduceScatterForwardAllGatherBackward(torch.autograd.Function):
    """ReduceScatter in forward and AllGather in backward"""

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, comm_name_):
        ctx.dim = dim_
        ctx.comm_name = comm_name_
        ctx.shapes = shapes_
        return _reduce_scatter(input_, dim_, shapes_, comm_name_)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _gather(grad_output, ctx.dim, ctx.shapes, ctx.comm_name),
            None,
            None,
            None,
        )


class _AllGatherForwardReduceScatterBackward(torch.autograd.Function):
    """AllGather in forward and ReduceScatter in backward"""

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, comm_name_):
        ctx.dim = dim_
        ctx.comm_name = comm_name_
        ctx.shapes = shapes_
        return _gather(input_, dim_, shapes_, comm_name_)

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce_scatter(grad_output, ctx.dim, ctx.shapes, ctx.comm_name), None, None, None


class RollForwardReverseRollBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shifts, dims, comm_names):
        """
        Forward pass of distributed roll.
        """
        assert len(shifts) == len(dims) == len(comm_names)
        ctx.shifts = shifts
        ctx.dims = dims
        ctx.comm_names = comm_names
        for sh, dim, comm_name in zip(shifts, dims, comm_names):
            if sh != 0:
                x = _distributed_roll(x, shift=sh, shift_dim=dim, comm_name=comm_name)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of distributed roll - just do a reverse roll.
        """
        shifts, dims, comm_names = ctx.shifts, ctx.dims, ctx.comm_names
        result = grad_output
        for sh, dim, comm_name in zip(shifts, dims, comm_names):
            if sh != 0:
                result = _distributed_roll(
                    result, shift=-sh, shift_dim=dim, comm_name=comm_name
                )
        return (
            result,
            None,
            None,
            None,
            None,
        )

class TransposeForwardTransposeBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dims, dim0_split_sizes, dim1_split_sizes, comm_name):
        # WAR for a potential contig check torch bug for channels last contig tensors
        x = _transpose(x, dims[0], dims[1], dim0_split_sizes, dim1_split_sizes, comm_name)
        ctx.dims = dims 
        ctx.dim0_split_sizes = dim0_split_sizes
        ctx.dim1_split_sizes = dim1_split_sizes
        ctx.comm_name = comm_name
        return x

    @staticmethod
    def backward(ctx, go):
        dims = ctx.dims
        dim0_split_sizes = ctx.dim0_split_sizes
        dim1_split_sizes = ctx.dim1_split_sizes
        comm_name = ctx.comm_name
        # WAR for a potential contig check torch bug for channels last contig tensors 
        gi = _transpose(go, dims[1], dims[0], dim1_split_sizes, dim0_split_sizes, comm_name)
        return gi, None, None, None, None   

# wrap autograd funcs in helpers
@profile("identity-allreduce")
def identity_forward_allreduce_backward(input_, comm_name):
    return _IdentityForwardAllReduceBackward.apply(input_, comm_name)


@profile("allreduce-identity")
def allreduce_forward_identity_backward(input_, comm_name):
    return _AllReduceForwardIdentityBackward.apply(input_, comm_name)


@profile("allgather-split")
def allgather_forward_split_backward(input_, dim, shapes, comm_name):
    return _AllGatherForwardSplitBackward.apply(input_, dim, shapes, comm_name)


@profile("allgather-reducescatter")
def allgather_forward_reducescatter_backward(input_, dim, shapes, comm_name):
    return _AllGatherForwardReduceScatterBackward.apply(input_, dim, shapes, comm_name)


@profile("reducescatter-allgather")
def reducescatter_forward_allgather_backward(input_, dim, shapes, comm_name):
    return _ReduceScatterForwardAllGatherBackward.apply(input_, dim, shapes, comm_name)


@profile("split-allgather")
def split_forward_allgather_backward(input_, dim, shapes, comm_name):
    return _SplitForwardAllGatherBackward.apply(input_, dim, shapes, comm_name)


@profile("roll")
def roll_forward_reverseroll_backward(x, shifts, dims, comm_names):
    return RollForwardReverseRollBackward.apply(x, shifts, dims, comm_names)


@profile("transpose-transpose")
def transpose_forward_transpose_backward(x, dims, dim0_split_sizes, dim1_split_sizes, comm_name):
    return TransposeForwardTransposeBackward.apply(x, dims, dim0_split_sizes, dim1_split_sizes, comm_name)


def init_ddp_model_and_reduction_hooks(
    model,
    device_ids,
    output_device,
    backend: str,
    bucket_cap_mb=25,
    broadcast_buffers=False,
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
    static_graph=False,
    verbose=False,
):
    # gloo backend initialization requires these to be None
    if backend == "gloo":
        device_ids = None
        output_device = None

    if comm.get_size("tp-sp1-sp2") == 1:
        # no model parallel, just use DDP with
        # the full world size
        need_hooks = False
        ddp_group = comm.get_group("dp") # this is world size here
    else:
        all_sp_reduce = True
        # if every param comm_metadata reduce group is sp1-sp2
        # then all wts need to be reduced in sp1-sp2
        # just do dp-sp1-sp2 reduces: careful about the scaling though
        for p in model.parameters():
            reduce_list = p.comm_metadata["reduce"]
            if not reduce_list or any(group != "sp1-sp2" for group in reduce_list):
                if (
                    reduce_list
                    and all(group == "tp-sp1-sp2" for group in reduce_list)
                    and comm.get_size("tp") == 1
                ):
                    # effectively only sp1-sp2
                    # WAR: for qk norm which needs a tp-sp1-sp2 reduce, ignore it if tp is 1
                    continue
                all_sp_reduce = False
                break
        ddp_group = comm.get_group("dp")
        need_hooks = True # need a grad hook for additional reduce or scaling

    if verbose and comm.get_world_rank() == 0:
        print(f"DDP model has all spatial parallel reduce: {all_sp_reduce}")

    model = DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device,
        bucket_cap_mb=bucket_cap_mb,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
        process_group=ddp_group,
    )

    if not need_hooks:
        return model

    # define comm hook because some params need additional allreduce
    def reduction_comm_hook(
        state: object, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        # allreduce everything first
        buff = bucket.buffer()
        # get future for allreduce

        with profile_range("async_allreduce"):
            # do the normal DDP all reduce
            grp = comm.get_group("dp-sp1-sp2") if all_sp_reduce else comm.get_group("dp")
            fut = dist.all_reduce(
                buff, op=dist.ReduceOp.SUM, group=grp, async_op=True
            ).get_future().then(lambda f: buff.div_(comm.get_size("dp")))

        if not all_sp_reduce:
            # need some additional allreduce hooks
            params = bucket.parameters()

            def grad_reduction(fut, grads, group):
                with profile_range(f"{group}_allreduce"):
                    coalesced = _flatten_dense_tensors(grads)
                    # extra allreduce for param wgrads that need it
                    dist.all_reduce(
                        coalesced,
                        op=dist.ReduceOp.SUM,
                        group=comm.get_group(group),
                        async_op=False,
                    )
                    for buf, synced in zip(
                        grads, _unflatten_dense_tensors(coalesced, grads)
                    ):
                        buf.copy_(synced)
                return bucket.buffer()

            # need more hooks
            for group in comm.get_names():
                if group == "dp":
                    continue
                grads = []
                for p in params:
                    # p needs an allreduce in group
                    if group in p.comm_metadata["reduce"]:
                        if p.grad is not None:
                            grads.append(p.grad.data)
                if not grads:
                    continue
                # append the new reduction functions
                fut = fut.then(partial(grad_reduction, grads=grads, group=group))

        return fut

        # if not append_hooks:
        #     # this bucket's params only needed the first allreduce
        #     # return the bucket directly
        #     return fut.then(lambda fut: fut.value()[0])
        # else:
        #     # got some additional allreduce chained to fut
        #     # the grad_reduction will return the bucket
        #     return fut

    # register model comm hook
    model.register_comm_hook(state=None, hook=reduction_comm_hook)
    return model
