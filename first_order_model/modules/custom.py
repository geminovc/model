import torch
from torch.nn import Conv2d as torch_conv2
from torch import Tensor
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
from typing import Union

class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        padding: Union[str, _size_2_t] = 0,
        groups: int = 1,
    ) -> None:
        kernel_size_ = _pair(kernel_size)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        super(Conv2d, self).__init__()
        self.depth_conv = torch_conv2(in_channels=in_channels, out_channels=in_channels,
            kernel_size=kernel_size_, stride=(1, 1), padding= padding, groups=in_channels)
        self.point_conv = torch_conv2(in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, stride=(1, 1))
        self.depthwise_separable_conv = torch.nn.Sequential(self.depth_conv, self.point_conv)

    def forward(self, input: Tensor) -> Tensor:
        return self.depthwise_separable_conv(input)
