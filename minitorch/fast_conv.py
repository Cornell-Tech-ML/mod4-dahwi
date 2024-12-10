from typing import Tuple, TypeVar, Any
import numpy as np
from numba import njit as _njit
from numba import prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """
    A decorator that applies the Numba `njit` (no Python) compilation to a function.
    This decorator uses Numba's `njit` to compile the given function to machine code,
    which can significantly improve performance by removing the Python interpreter overhead.
    Args:
        fn (Fn): The function to be compiled.
        **kwargs (Any): Additional keyword arguments to pass to the Numba `njit` function.
    Returns:
        Fn: The compiled function.
    """

    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation with Numba.

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right.

    """
    batch, out_channels, out_width = out_shape
    batch_, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    for b in prange(batch):
        for oc in prange(out_channels):
            for w in prange(out_width):
                acc = 0.0
                for ic in range(in_channels):
                    for k in range(kw):
                        # Compute the input position based on kernel alignment
                        input_pos = w - k if reverse else w + k
                        if 0 <= input_pos < width:
                            input_idx = b * s1[0] + ic * s1[1] + input_pos * s1[2]
                            weight_idx = oc * s2[0] + ic * s2[1] + k * s2[2]
                            acc += input[input_idx] * weight[weight_idx]
                # Write result to output
                out_idx = b * out_strides[0] + oc * out_strides[1] + w * out_strides[2]
                out[out_idx] = acc


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes the gradients of the input and weight tensors for a 1D convolution operation during the backward pass.
        Args:
            ctx (Context): The context object that stores information from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output of the convolution.
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the gradients of the loss with respect to the input and weight tensors.
        """

        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    # Iterate over the output tensor
    for i in prange(out_size):
        # Get the current out_index based on out_shape
        out_index = np.empty(4, np.int32)
        to_index(i, out_shape, out_index)
        current_batch, current_out_channel, current_out_height, current_out_width = out_index

        # Accumulator for the convolution
        acc = 0.0

        # Iterate through the kernel
        for current_in_channel in range(in_channels):
            for kernel_height in range(kh):
                for kernel_width in range(kw):
                    # Current offset in convolution (anchor right if reverse)
                    conv_offset_h = (kh - 1 - kernel_height) if reverse else kernel_height
                    conv_offset_w = (kw - 1 - kernel_width) if reverse else kernel_width

                    # Current weight value
                    weight_pos = (
                        current_out_channel * weight_strides[0]
                        + current_in_channel * weight_strides[1]
                        + conv_offset_h * weight_strides[2]
                        + conv_offset_w * weight_strides[3]
                    )

                    # Current input value (subtract offset if reverse)
                    input_height = (
                        current_out_height - conv_offset_h
                        if reverse
                        else current_out_height + conv_offset_h
                    )
                    input_width = (
                        current_out_width - conv_offset_w
                        if reverse
                        else current_out_width + conv_offset_w
                    )

                    # Check if input is in bounds
                    if 0 <= input_height < height and 0 <= input_width < width:
                        input_pos = (
                            current_batch * input_strides[0]
                            + current_in_channel * input_strides[1]
                            + input_height * input_strides[2]
                            + input_width * input_strides[3]
                        )
                        acc += input[input_pos] * weight[weight_pos]

        # Write the result to the output tensor
        out_pos = index_to_position(out_index, out_strides)
        out[out_pos] = acc

tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes the gradient of the loss with respect to the input and weight tensors
        during the backward pass of the convolution operation.
        Args:
            ctx (Context): The context object that contains saved values from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output of the convolution.
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the gradients of the loss with respect to the input
            and weight tensors, respectively.
        """

        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
