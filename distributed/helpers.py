import torch
import torch.distributed as dist
from utils import comm


def init_params_for_shared_weights(model):
    """Helper routine to ensure shared weights are the same after initialization"""
    with torch.no_grad():
        # distributed sync step
        for param in model.parameters():
            if not hasattr(param, "comm_metadata"):
                # if you forget to annotate, then it is a shared weight
                param.comm_metadata = {
                    "sharded": [],
                    "shared": ["tp-sp1-sp2"],
                    "reduce": [],
                }

            for comm_group in param.comm_metadata["shared"]:
                if comm.get_size(comm_group) > 1:
                    # broadcast from rank 0 to ensure all ranks have the same shared weights
                    dist.broadcast(
                        param,
                        src=comm.get_root(comm_group),
                        group=comm.get_group(comm_group),
                    )


# distributed primitives
# helper routine to compute uneven splitting in balanced way:
def compute_split_shapes(size, num_chunks):
    if num_chunks == 1:
        return [size]
    
    # first, check if we can split using div-up to balance the load:
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        # in this case, the last shard would be empty, split with floor instead:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks - 1)

    # generate sections list
    sections = [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]

    return sections


# need a separate helper for patching to ensure distr and seq setting are consistent
# for example, 721 split 4 ways would give 181, 181, 181, 178 which is valid, but if
# you now patched each chunk we would lose pixels in all chunks. instead, it should
# have been 180, 180, 180, 181 which is valid and consistent with the sequential setting
# so first divide by patch size, then split, then multiply by patch size
def compute_split_shapes_for_patching(size, num_chunks, patch_size):
    result = compute_split_shapes(size // patch_size, num_chunks)

    if patch_size == 1:
        return result

    # result is in patches, bring back to pixels
    result = [ s * patch_size for s in result ]

    # add the remainder to the last element
    result[-1] += size % patch_size

    return result


def _reduce(input_, comm_name):
    """All-reduce the input tensor across model parallel group."""
    # Bypass the function if we are using only 1 GPU or if
    # communicator is not initialized
    if comm.get_size(comm_name) == 1:
        return input_

    # All-reduce.
    input_ = input_.contiguous()
    dist.all_reduce(input_, group=comm.get_group(comm_name))

    return input_


def split_tensor_along_dim(tensor, dim, num_chunks, sections):
    """Helper routine to split a tensor along a given dimension"""
    if dim >= tensor.dim():  # scattering from dim that doesnt exist
        raise ValueError(
            f"Error: Scattering along {dim} for a tensor of size {tensor.dim()}"
        )
    if tensor.shape[dim] < num_chunks:
        raise ValueError(
            f"Error, cannot split dim {dim} of size {tensor.shape[dim]} into {num_chunks} chunks"
        )

    # get split
    if sections is None:
        sections = compute_split_shapes(tensor.shape[dim], num_chunks)
    tensor_list = list(torch.split(tensor, sections, dim=dim))

    return tensor_list


def _split(input_, dim_, comm_name, sections):
    """Split the tensor along dim."""
    # Bypass the function if we are using only 1 GPU or if
    # communicator is not initialized
    comm_size = comm.get_size(comm_name)
    if comm_size == 1:
        return input_

    # Split along  dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size, sections)

    # Note: torch.split does not create contiguous tensors by default.
    comm_rank = comm.get_rank(comm_name)
    output = input_list[comm_rank].contiguous()

    return output


def _gather(input_, dim_, shapes_, comm_name):
    """
    Gather tensors and concatinate along the dimension dim_.
    """
    comm_size = comm.get_size(comm_name)
    if (shapes_ is not None) and (len(shapes_) != comm_size):
        raise ValueError(f"Error: passed shapes of size not equal to {comm_size}")
    if dim_ >= input_.dim():  # gathering along dim that doesnt exist
        raise ValueError(
            f"Error: Gathering along {dim_} for a tensor of size {input_.dim()}"
        )

    # Bypass the function if we are using only 1 GPU or if
    # communicator is not initialized
    if comm_size == 1:
        return input_

    input_ = input_.contiguous()
    input_shape = list(input_.shape)
    if shapes_ is not None:
        input_list = []
        for src in range(comm_size):
            input_shape[dim_] = shapes_[src]
            input_list.append(
                torch.empty(input_shape, dtype=input_.dtype, device=input_.device)
            )
    else:
        # assume equal shape on all ranks
        input_list = [torch.empty_like(input_) for _ in range(comm_size)]

    dist.all_gather(input_list, input_, group=comm.get_group(comm_name))
    output = torch.cat(input_list, dim=dim_).contiguous()

    return output


def _reduce_scatter(input_, dim_, shapes_, comm_name):
    """
    Reduces and scatters along dim_
    """
    comm_size = comm.get_size(comm_name)
    if dim_ >= input_.dim():  # RS along dim that doesnt exist
        raise ValueError(
            f"Error: Reduce-scatter along {dim_} for a tensor of size {input_.dim()}"
        )

    # Bypass the function if we are using only 1 GPU or if
    # communicator is not initialized
    if comm_size == 1:
        return input_

    comm_rank = comm.get_rank(comm_name)
    input_ = input_.contiguous()

    # Split along  dimension. Make sure the individual tensors are contiguous!
    input_list = [
        t.contiguous() for t in split_tensor_along_dim(input_, dim_, comm_size, shapes_)
    ]

    output = torch.empty_like(input_list[comm_rank].contiguous())
    dist.reduce_scatter(output, input_list, group=comm.get_group(comm_name))

    return output


def invert_permutation(sigma):
    inv = [0] * len(sigma)
    for i, p in enumerate(sigma):
        inv[p] = i
    return inv


def _distributed_roll(x, shift: int, shift_dim: int, comm_name: str):
    """
    Implementation of distributed roll on a tensor split along one axis using all_to_all_single.

    Args:
    - x: Tensor of shape (B, ..., H, W, C) on each GPU where ... can be any number of dimensions
    - shift: shift in the dimension
    - shift_dim: dimension of shift
    - comm_name: Name of the communication group

    Returns:
    - Rolled tensor of shape (B, ..., H, W, C) on each GPU
    """
    if shift_dim < 0:
        shift_dim = x.dim() + shift_dim
    rank = comm.get_rank(comm_name)
    size = comm.get_size(comm_name)

    # non-distributed case
    if size == 1:
        return torch.roll(x, shifts=(shift,), dims=(shift_dim,))

    group = comm.get_group(comm_name)

    # either send right or left
    send_right = max(0, shift)
    send_left = max(0, -shift)

    if send_right == 0 and send_left == 0:
        return x

    # get neighbor ranks
    right_neighbor = (rank + 1) % size
    left_neighbor = (rank - 1) % size

    # move the dimension we want to shift to the front,
    # save the remaining shape to use in buffer
    remainder_shape = [x.shape[i] for i in range(x.ndim) if i != shift_dim]
    permute_dims = [shift_dim] + list([i for i in range(x.ndim) if i != shift_dim])
    send_inds = [slice(None)] * x.ndim
    if send_right > 0:
        send_inds[shift_dim] = slice(-send_right, None)
        recv_shape = [send_right] + remainder_shape
    elif send_left > 0:
        send_inds[shift_dim] = slice(0, send_left)
        recv_shape = [send_left] + remainder_shape
    # permute because all_to_all_single comms in dim=0 only(?)
    # move shift dimension to front while preserving batch dimensions
    send_inds = tuple(send_inds)
    send_buf = x[send_inds].permute(permute_dims).contiguous()
    recv_buf = torch.empty(recv_shape, device=x.device, dtype=x.dtype)

    # define send/recv sizes
    input_split_sizes = [0] * size
    output_split_sizes = [0] * size

    if send_right > 0:
        input_split_sizes[right_neighbor] = send_buf.shape[0]  # dim=0 now
        output_split_sizes[left_neighbor] = recv_buf.shape[0]
    elif send_left > 0:
        input_split_sizes[left_neighbor] = send_buf.shape[0]
        output_split_sizes[right_neighbor] = recv_buf.shape[0]

    # collective communication
    dist.all_to_all_single(
        recv_buf, send_buf, output_split_sizes, input_split_sizes, group=group
    )

    inverse_permute = invert_permutation(permute_dims)
    recv_buf = recv_buf.permute(inverse_permute).contiguous()

    # local roll
    x = torch.roll(x, shifts=(shift,), dims=(shift_dim,))

    # reconstruct the tensor
    keep_slice = [slice(None, None)] * x.ndim
    if send_right > 0:  # received from left
        keep_slice[shift_dim] = slice(send_right, None)
        keep_slice = tuple(keep_slice)
        x = torch.cat([recv_buf, x[keep_slice]], dim=shift_dim)
    elif send_left > 0:  # received from right
        keep_slice[shift_dim] = slice(None, -send_left)
        keep_slice = tuple(keep_slice)
        x = torch.cat([x[keep_slice], recv_buf], dim=shift_dim)

    return x

def _transpose(tensor, dim0, dim1, dim0_split_sizes, dim1_split_sizes, comm_name, async_op=False): 
    # get comm params
    comm_size = comm.get_size(comm_name)
    comm_rank = comm.get_rank(comm_name)

    # split and local transposition
    tsplit = split_tensor_along_dim(tensor, num_chunks=comm_size, dim=dim0, sections=dim0_split_sizes)
    x_send = [y.contiguous() for y in tsplit]
    x_send_shapes = [x.shape for x in x_send]
    x_recv = []
    x_shape = list(x_send_shapes[comm_rank])
    for dim1_len in dim1_split_sizes:
        x_shape[dim1] = dim1_len
        x_recv.append(torch.empty(x_shape, dtype=tensor.dtype, device=tensor.device))
        
    # global transposition
    dist.all_to_all(x_recv, x_send, group=comm.get_group(comm_name), async_op=async_op)

    return torch.cat(x_recv, dim=dim1)


