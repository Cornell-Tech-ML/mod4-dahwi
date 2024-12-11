# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Uses the Just-In-Time (JIT) compilation to optimize the given function
    for execution on a specific device, such as a GPU. It leverages the `_jit` function
    with the `device` parameter set to `True`.

    Args:
    ----
        fn (Fn): The function to be JIT compiled.
        **kwargs: Additional keyword arguments to be passed to the `_jit` function.

    Returns:
    -------
        Fn: The JIT compiled function optimized for the specified device.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Any, **kwargs: Any) -> FakeCUDAKernel:
    """Just-In-Time (JIT) compilation decorator for CUDA operations.
    This function takes a function `fn` and additional keyword arguments,
    and returns a `FakeCUDAKernel` object that represents the JIT-compiled
    version of the function.

    Args:
    ----
        fn (Callable): The function to be JIT-compiled.
        **kwargs: Additional keyword arguments to be passed to the JIT compiler.

    Returns:
    -------
        FakeCUDAKernel: The JIT-compiled version of the input function.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            to_index(i, out_shape, out_index)

            # Use broadcasting to map the output index to the input index
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Calculate the position in input and output storage
            in_pos = index_to_position(in_index, in_strides)
            # out_pos = index_to_position(out_index, out_strides)

            # Apply the function and store the result in the output
            out[i] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, Storage, Shape, Strides],
    None,
]:
    r"""CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            # out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            out[i] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0

    cuda.syncthreads()
    s = 1
    while s < BLOCK_DIM:
        if pos % (2 * s) == 0 and pos + s < BLOCK_DIM:
            cache[pos] += cache[pos + s]
        s *= 2
        cuda.syncthreads()

    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Computes the sum of elements in the given tensor `a` using CUDA parallelization.

    Args:
    ----
        a (Tensor): The input tensor whose elements are to be summed.

    Returns:
    -------
        TensorData: A tensor containing the sum of the elements in `a`.

    Notes:
    -----
        - The function uses CUDA for parallel computation.
        - The result is stored in a TensorData object with a shape of (2,).
        - The function assumes that the input tensor `a` is already on the GPU.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, int, Storage, Shape, Strides, int, float], None
]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        if out_pos < out_size:
            # Initialize cache with reduce_value
            cache[pos] = reduce_value
            # starting index in out_shape
            to_index(out_pos, out_shape, out_index)
            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos
            start = index_to_position(out_index, a_strides)
            # Combine reduce_value with a_storage[start] if within bounds
            if out_index[reduce_dim] < a_shape[reduce_dim]:
                if pos == 0:
                    cache[pos] = fn(cache[pos], a_storage[start])
                else:
                    cache[pos] = a_storage[start]
                cuda.syncthreads()

                s = 1
                while s < BLOCK_DIM:
                    if pos % (2 * s) == 0 and pos + s < BLOCK_DIM:
                        cache[pos] = fn(cache[pos], cache[pos + s])
                        cuda.syncthreads()
                    s *= 2

            if pos == 0:
                out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    r"""Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y

    pos = i * size + j
    if i < size and j < size:
        a_shared[local_i, local_j] = a[pos]
        b_shared[local_i, local_j] = b[pos]
        cuda.syncthreads()

        acc = 0
        for k in range(size):
            acc += a_shared[local_i, k] * b_shared[k, local_j]
        out[pos] = acc


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Perform matrix multiplication of two tensors using CUDA. The result tensor is
        initialized on the GPU and the matrix multiplication is performed using a
        CUDA kernel function `jit_mm_practice`.

    Args:
    ----
        a (Tensor): The first input tensor with shape (size, size).
        b (Tensor): The second input tensor with shape (size, size).

    Returns:
    -------
        TensorData: The result of the matrix multiplication as a TensorData object.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Determine the batch stride for the output tensor
    out_batch_stride = out_strides[0] if out_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    # Define the position in the output storage
    pos = batch * out_batch_stride + i * out_strides[-2] + j * out_strides[-1]
    # The local position in the block.
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    acc = 0.0  # Initialize the accumulator for the dot product

    # Iterate over the shared dimension (a_shape[-1]) by block dim
    for k in range(0, a_shape[-1], BLOCK_DIM):
        # Load a block of data from a_storage into shared memory at the current thread position
        # Guarding: Ensure that both the row index (`i`) is within the valid range of `a_shape[-2]`
        # (number of rows in matrix `a`) and the column index (`k + local_j`) is within the valid
        # range of `a_shape[-1]` (number of columns in matrix `a`). Without this check, threads
        # outside these bounds would try to access invalid memory locations.
        if i < a_shape[-2] and k + local_j < a_shape[-1]:
            a_shared[local_i, local_j] = a_storage[
                batch * a_batch_stride
                + i * a_strides[-2]
                + (k + local_j) * a_strides[-1]
            ]
        # Load a block of data from b_storage into shared memory
        # Guarding: Ensure that the column index (`j`) is within the valid range of `b_shape[-1]`
        # (number of columns in matrix `b`) and the row index (`k + local_i`) is within the valid
        # range of `b_shape[-2]` (number of rows in matrix `b`). This ensures that threads don't
        # read beyond valid memory regions of `b_storage`.
        if j < b_shape[-1] and k + local_i < b_shape[-2]:
            b_shared[local_i, local_j] = b_storage[
                batch * b_batch_stride
                + (k + local_i) * b_strides[-2]
                + j * b_strides[-1]
            ]
        cuda.syncthreads()  # Synchronize threads to ensure shared memory is fully loaded

        # Compute the dot product for the current block
        # Guarding: Ensure the indices within the shared memory (`local_k`) are valid for both the
        # shared dimension of matrix `a` (columns) and matrix `b` (rows). This guarantees that only
        # valid elements of `a_shared` and `b_shared` are accessed during the dot product computation.
        for local_k in range(BLOCK_DIM):
            if k + local_k < a_shape[-1] and k + local_k < b_shape[-2]:
                acc += a_shared[local_i, local_k] * b_shared[local_k, local_j]

    # Write the computed value to the output storage
    if i < out_shape[-2] and j < out_shape[-1] and pos < out_size:
        out[pos] = acc


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
