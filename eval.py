import os

from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader

from core.EDSR.edsr import EDSR
from core.option import parser
from core.model import WDSR_A, WDSR_B, WDSR_Deconv, WDSR_Norm_Deconv
from core.data.div2k import DIV2K
from core.data.utils import quantize
from core.utils import AverageMeter, calc_psnr, load_checkpoint, load_weights
from PIL import Image
import numpy as np
from core.data.dir_dataset import DirDataSet


def forward(x, model):
    with torch.no_grad():
        sr = model(x)
        return sr


def forward_x8(x, model):
    x = x.squeeze(0).permute(1, 2, 0)

    with torch.no_grad():
        sr = []

        for rot in range(0, 4):
            for flip in [False, True]:
                _x = x.flip([1]) if flip else x
                _x = _x.rot90(rot)
                out = model(_x.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
                out = out.rot90(4 - rot)
                out = out.flip([1]) if flip else out
                sr.append(out)

        return torch.stack(sr).mean(0).permute(2, 0, 1)


def test(dataset, loader, model, args, device, tag=''):
    psnr = AverageMeter()

    # Set the model to evaluation mode
    model.eval()

    with tqdm(total=len(dataset)) as t:
        t.set_description(tag)

        count = 0
        for data in loader:
            lr, hr = data
            lr = lr.to(device)
            hr = hr.to(device)

            if args.self_ensemble:
                sr = forward_x8(lr, model)
            else:
                sr = forward(lr, model)

            # Quantize results
            sr = quantize(sr, args.rgb_range)

            if args.output_dir:
                im = Image.fromarray(np.uint8(sr.squeeze().cpu().numpy().transpose(2, 1, 0))).transpose(Image.ROTATE_270)
                im_lr = Image.fromarray(np.uint8(lr.squeeze().cpu().numpy().transpose(2, 1, 0))).transpose(Image.ROTATE_270)
                im.save(args.output_dir + str(count) + ".png")
                im_lr.save(args.output_dir + str(count) + "_lr.png")
                count += 1

            # to save activation layers
            # act = activation['deconv'].squeeze()
            #
            # for idx in range(50):
            #     plt.imsave('results/output/act/activation%d_.png' % idx, act[idx], vmin=act[idx].min(), vmax=act[idx].max())

            # Update PSNR
            # psnr.update(calc_psnr(sr, hr, scale=args.scale, max_value=args.rgb_range[1]), lr.shape[0])

            t.update(lr.shape[0])

    print('DIV2K (val) PSNR: {:.4f} dB'.format(psnr.avg))

# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook

if __name__ == '__main__':
    # Define specific options and parse arguments
    parser.add_argument('--dataset-dir', type=str, required=True, help='DIV2K Dataset Root Directory')
    parser.add_argument('--checkpoint-file', type=str, required=True)
    parser.add_argument('--self-ensemble', action='store_true')
    parser.add_argument('--output-dir', type=str, required=False)
    args = parser.parse_args()

    # Set cuDNN auto-tuner and get device
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.output_dir is not None and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create model
    if args.model == 'WDSR-B':
        model = WDSR_B(args).to(device)
    elif args.model == 'WDSR-A':
        model = WDSR_A(args).to(device)
    elif args.model == 'EDSR':
        model = EDSR(args).to(device)
    elif args.model == 'EDSR-Deconv':
        args.deconv = True
        model = EDSR(args).to(device)
    elif args.model == 'EDSR-Norm-Deconv':
        args.deconv = True
        args.norm = True
        model = EDSR(args).to(device)
    elif args.model == 'WDSR-Deconv':
        model = WDSR_Deconv(args).to(device)
    else:
        model = WDSR_Norm_Deconv(args).to(device)

    # Load weights
    model = load_weights(model, load_checkpoint(args.checkpoint_file)['state_dict'])

    # to see activation layers
    # model.tail._modules['0'].register_forward_hook(get_activation('deconv'))

    # Prepare dataset
    dataset = DirDataSet('data/checkers')
    # dataset = DIV2K(args, train=False)
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    test(dataset, dataloader, model, args, device)
