import torch
from torch.nn import Conv2d as torch_conv2
from torch import Tensor
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
from typing import Union

class Conv2d(torch.nn.Module):
    """
    This module is used to replce a regular convolutions with depthwise separable convolutions.
    Depthwise separable convolutions yield the same performance while being faster.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        padding: Union[str, _size_2_t] = 0,
        groups: int = 1,
        bias=True
    ) -> None:

        kernel_size_ = _pair(kernel_size)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        super(Conv2d, self).__init__()
        self.depth_conv = torch_conv2(in_channels=in_channels, out_channels=in_channels,
            kernel_size=kernel_size_, stride=(1, 1), padding=padding, groups=in_channels, bias=bias)
        self.point_conv = torch_conv2(in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, stride=(1, 1), bias=bias)
        self.depthwise_separable_conv = torch.nn.Sequential(self.depth_conv, self.point_conv)

    def forward(self, input: Tensor) -> Tensor:
        return self.depthwise_separable_conv(input)

    def zero_out_weight(self):
        self.depth_conv.weight.data.zero_()
        self.point_conv.weight.data.zero_()

    def adjust_bias(self, in_channels, num_maps):
        self.depth_conv.bias.data.copy_(torch.ones(in_channels, dtype=torch.float))
        self.point_conv.bias.data.copy_(torch.tensor([1, 0, 0, 1] * num_maps, dtype=torch.float))

