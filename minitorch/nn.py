from typing import Tuple

from .tensor import Tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    # Reshape and permute to group kernel elements
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    permuted = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()  # (batch, channel, new_height, new_width, kh, kw)

    tiled = permuted.view(batch, channel, new_height, new_width, kh * kw)
    
    return tiled, new_height, new_width

def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to the input tensor.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    tiled, new_height, new_width = tile(input, kernel)

    res = tiled.mean(dim=4)  # Average over kernel elements

    return res.view(input.shape[0], input.shape[1], new_height, new_width)