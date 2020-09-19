from typing import Union, Tuple
from torch import nn
import torch

eps = 1e-10
T = int


class NormConvTranspose2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]],
                 stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0,
                 output_padding: Union[T, Tuple[T, T]] = 0, groups: int = 1, bias: bool = True, dilation: int = 1,
                 padding_mode: str = 'zeros'):
        super().__init__()

        if groups != 1:
            raise NotImplementedError('NormConvTranspose2d is currently not implemented from groups != 1')

        self.out_channels = out_channels

        if bias:
            self._bias = nn.Parameter(torch.zeros(out_channels))

        self.sub_convs = []
        for i in range(self.out_channels):
            self.sub_convs.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride, padding,
                                                     output_padding=output_padding, groups=in_channels, bias=False,
                                                     dilation=dilation, padding_mode=padding_mode))

    def forward(self, input, output_size=None):

        output = None
        ones = torch.ones_like(input)
        for i in range(self.out_channels):
            x = self.sub_convs[i](input)
            normalizer = self.sub_convs[i].forward(ones)
            if output is None:
                output = torch.Tensor(x.shape[0], self.out_channels, x.shape[2], x.shape[3])
            output[:, i, :, :] = torch.sum(x / normalizer, dim=1)
            if self._bias is not None:
                output[:, i, :, :] += self._bias[i]

        return output

    def fill(self, tensor):

        for i in range(self.out_channels):
            self.sub_convs[i].weight.data = tensor[i].unsqueeze(1)
            
    def to(self, *args, **kwargs):
        super(NormConvTranspose2d, self).to(*args, **kwargs)
        for conv in self.sub_convs:
            conv.to(*args, **kwargs)
