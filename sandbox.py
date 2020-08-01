import numpy as np
from torch import nn
from core.data.dir_dataset import DirDataSet
import torch
from core.model.common import ShiftMean
from matplotlib import pyplot as plt
from core.model.NormConvTranspose2d import NormConvTranspose2d
import plotly.graph_objects as go


# create deconv
w = np.load('deconv_w.npy')
b = np.load('deconv_b.npy')
b_sum = np.sum(b)
deconv = nn.ConvTranspose2d(3, 3, kernel_size=5, stride=4, padding=2, output_padding=3)

deconv_inv = nn.ConvTranspose2d(3, 3, kernel_size=5, stride=4, padding=2, output_padding=3)
deconv_one = nn.ConvTranspose2d(3, 3, kernel_size=5, stride=4, padding=2, output_padding=3)
deconv_one_n = NormConvTranspose2d(3, 3, kernel_size=5, stride=4, padding=2, output_padding=3)


deconv.weight.data = torch.Tensor(w)
deconv.bias.data = torch.Tensor(b)

deconv_inv.weight.data = torch.Tensor(w[:, ::-1,::-1,::-1].copy())
deconv_inv.bias.data = torch.Tensor(b)

deconv_one_n.transpose2d.weight.data.fill_(0.1)
deconv_one_n.transpose2d.bias.data.fill_(0)

deconv_one.weight.data.fill_(0.1)
deconv_one.bias.data.fill_(0)
# deconv_one.bias.data = torch.Tensor(b)

# get image
shift = ShiftMean([0.4488, 0.4371, 0.4040])
dataset = DirDataSet('data/checkers')
image_small, image_big = dataset.__getitem__(0)
image_small = image_small.unsqueeze(0)
image = shift(image_small, mode='sub')

######################################

ones_ = torch.ones_like(image)
norm = deconv_one(ones_)
norm_b = deconv_one(ones_)
# infer
res = deconv_one_n(image)
b_tensor = torch.Tensor(b.reshape(1, 3, 1, 1).repeat(res.shape[2], axis=2).repeat(res.shape[3], axis=3))
# res -= b_tensor
# res /= norm
# res += b_tensor

######################################

# show result
res = shift(res, mode='add')
res = res.detach().numpy().squeeze().transpose(2,1,0)/255.
plt.imshow(res, interpolation='nearest')

focus = res[600:700, 1000:1200]
plt.imshow(focus, interpolation='nearest')
plt.show()
# plt.imshow(norm_b.detach().numpy().squeeze().transpose(2,1,0)[600:700, 1000:1200], interpolation='nearest')

plt.show()
# fig = go.Figure().add_image(customdata=res)
# fig.show()
print('s')
