from torch import nn
import torch

class NormConvTranspose2d(nn.Module):

    def __init__(self,  *args, **kwargs):
        super(NormConvTranspose2d, self).__init__()
        self.transpose2d = nn.ConvTranspose2d(*args, **kwargs)

    def forward(self, x):
        ones = torch.ones_like(x)
        ones.requires_grad = False
        x = self.transpose2d(x)
        normalizer = self.transpose2d(ones)
        x = x/normalizer

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
