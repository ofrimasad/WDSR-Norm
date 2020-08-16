from torch import nn
import torch

eps = 1e-10
class NormConvTranspose2d(nn.ConvTranspose2d):

    def forward(self, input, output_size=None):
        ones = torch.ones_like(input)
        ones.requires_grad = False
        x = super().forward(input, output_size)
        normalizer = super().forward(ones, output_size)
        unsqueezed_bias = self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        if normalizer.nonzero().size(0) < normalizer.numel():
            normalizer += eps

        x += unsqueezed_bias
        x /= normalizer
        x -= unsqueezed_bias

        return x

#
# class NormConvTranspose2d(nn.Module):
#
#     def __init__(self,  *args, **kwargs):
#         super(NormConvTranspose2d, self).__init__()
#         self.transpose2d = nn.ConvTranspose2d(*args, **kwargs)
#         self.transpose2d_for_norm = nn.ConvTranspose2d(*args, **kwargs)
#         self.transpose2d_for_norm.weight.data.fill_(1/kwargs['stride']**2)
#         self.was_init = False
#
#     def forward(self, x):
#         self.init_norm(x)
#         x = self.transpose2d(x)
#         x = x/self.normalizer
#         return x
#
#     def init_norm(self, x):
#         if not self.was_init:
#             self.was_init = True
#             ones = torch.ones_like(x)
#             ones.requires_grad = False
#             self.normalizer = self.transpose2d_for_norm(ones)
#

# class NormConvTranspose2d(nn.Module):
#
#     def __init__(self,  *args, **kwargs):
#         super(NormConvTranspose2d, self).__init__()
#         self.transpose2d = nn.ConvTranspose2d(*args, **kwargs)
#         self.transpose2d_for_norm = nn.ConvTranspose2d(*args, **kwargs)
#         self.transpose2d_for_norm.weight.data.fill_(1)
#         self.was_init = False
#
#     def forward(self, x):
#         self.init_norm(x)
#         ones = torch.ones_like(x)
#         ones.requires_grad = False
#         x = self.transpose2d(x)
#         normalizer = self.transpose2d(ones)
#         #normalizer /= self.normalizer
#         x = x/normalizer
#
#         return x
#
#     def init_norm(self, x):
#         #if not self.was_init:
#         self.was_init = True
#         ones = torch.ones_like(x)
#         ones.requires_grad = False
#         self.normalizer = self.transpose2d_for_norm(ones)
