from torch import nn
import torch

eps = 1e-10


class NormConvTranspose2d(nn.ConvTranspose2d):

    def forward(self, input, output_size=None):

        ones = torch.ones_like(input)
        x = super().forward(input, output_size)
        normalizer = super().forward(ones, output_size)
        unsqueezed_bias = self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # if normalizer.nonzero().size(0) < normalizer.numel():
        #     normalizer += eps

        x += unsqueezed_bias
        x /= normalizer
        x -= unsqueezed_bias

        return x
