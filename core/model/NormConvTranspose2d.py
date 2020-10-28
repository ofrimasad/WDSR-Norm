from typing import Union, Tuple, Optional
from torch import nn
import torch
from torch._C import device
import numpy as np
eps = 1e-10
T = int


class NormConvTranspose2d_n(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]],
                 stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0,
                 output_padding: Union[T, Tuple[T, T]] = 0, groups: int = 1, bias: bool = True, dilation: int = 1,
                 padding_mode: str = 'zeros'):
        super(NormConvTranspose2d_n, self).__init__()

        if groups != 1:
            raise NotImplementedError('NormConvTranspose2d is currently not implemented from groups != 1')

        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels))
        if bias:
            self._bias = nn.Parameter(torch.zeros(out_channels))

        self.sub_convs = []
        for i in range(self.out_channels):
            m = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride, padding,
                                                     output_padding=output_padding, groups=in_channels, bias=False,
                                                     dilation=dilation, padding_mode=padding_mode)
            self.sub_convs.append(m)
            self.add_module(str(i), m)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, input, output_size=None):

        output = None
        ones = torch.ones_like(input).to(input.device)
        for i in range(self.out_channels):
            w_input = input * self.weight[i].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            x = self.sub_convs[i](w_input)
            normalizer = self.sub_convs[i].forward(ones) + eps
            if output is None:
                output = torch.Tensor(x.shape[0], self.out_channels, x.shape[2], x.shape[3]).to(input.device)
            output[:, i, :, :] = torch.sum(x / normalizer, dim=1)
            if self._bias is not None:
                output[:, i, :, :] += self._bias[i]

        return output

    def fill(self, w, bias):

        for i in range(self.out_channels):
            self.sub_convs[i].weight.data = w[:,i].unsqueeze(1)
        self._bias.data = bias


class NormConvTranspose2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[T, Tuple[T, T]],
                 stride: Union[T, Tuple[T, T]] = 1, padding: Union[T, Tuple[T, T]] = 0,
                 output_padding: Union[T, Tuple[T, T]] = 0, groups: int = 1, bias: bool = True, dilation: int = 1,
                 padding_mode: str = 'zeros'):
        super(NormConvTranspose2d, self).__init__()

        if groups != 1:
            raise NotImplementedError('NormConvTranspose2d is currently not implemented from groups != 1')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.deconv = nn.ConvTranspose2d(in_channels, in_channels * out_channels, kernel_size, stride, padding,
                                                     output_padding=output_padding, groups=in_channels, bias=False,
                                                     dilation=dilation, padding_mode=padding_mode)
        self.weight = nn.Parameter(torch.rand(1, out_channels*in_channels, 1, 1))
        self.sum = nn.Conv2d(in_channels * out_channels, out_channels, kernel_size=1, groups=out_channels)
        self.sum.weight.data.fill_(1)
        self.sum.weight.requires_grad = False  # we train the sum bias only

        index = np.ndarray((in_channels * out_channels), dtype=np.long)
        for i in range(out_channels):
            index[i * in_channels: (i + 1) * in_channels] = np.arange(i, in_channels*out_channels, in_channels)
        self.index = nn.Parameter(torch.LongTensor(index), requires_grad=False)

    def forward(self, input, output_size=None):

        self.sum.weight.requires_grad = False  # we train the sum bias only

        normalizer = torch.ones_like(input).to(input.device)
        x = self.deconv(input)
        x *= self.weight
        normalizer = self.deconv(normalizer) + eps
        x /= normalizer
        x = torch.index_select(x, dim=1, index=self.index)
        x = self.sum(x)
        return x

    def fill(self, tensor, bias):

        self.deconv.weight.data = tensor
        self.sum.bias.data = bias


