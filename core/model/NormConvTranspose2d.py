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