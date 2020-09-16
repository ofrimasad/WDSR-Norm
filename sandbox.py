from torch import nn
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from core.data.dir_dataset import DirDataSet
from core.model import NormConvTranspose2d
from core.model.NormConvTranspose2d import NormConvTranspose2d

eps = 1e-10

# create image
img = torch.zeros((1, 3, 20, 20))
img[0][0] = 0.8
img[0][1] = 0.3
img[0][2] = 0.2

dataset = DirDataSet('images/sample')
dataloader = DataLoader(dataset=dataset, batch_size=1)
img = dataset.__getitem__(1)[0]

img = img.unsqueeze(0) / 255.0

plt.imshow(img.squeeze().permute(1, 2, 0))
deconv = nn.ConvTranspose2d(3, 3, kernel_size=5, stride=4, padding=2, output_padding=3)
norm_deconv = NormConvTranspose2d(3, 3, kernel_size=5, stride=4, padding=2, output_padding=3)

rand_w = torch.randn(3, 3, 5, 5)
deconv.weight.data = rand_w  # .fill_(0.4)
norm_deconv.fill(rand_w)
# norm_deconv_no_bias.weight.data = rand_w  # .fill_(0.4)

res_deconv = deconv(img)*0.2
res_deconv_norm = norm_deconv(img)*0.2

plt.imshow(res_deconv.detach().squeeze().permute(1, 2, 0), vmin=0, vmax=10, interpolation='none')  # .permute(1, 2, 0))
plt.show()

plt.imshow(res_deconv_norm.detach().squeeze().permute(1, 2, 0), vmin=0, vmax=10, interpolation='none')  # .permute(1, 2, 0))
plt.show()

print('done')
