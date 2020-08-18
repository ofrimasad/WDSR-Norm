from core.EDSR import common

import torch
import torch.nn as nn
from core.model.NormConvTranspose2d import NormConvTranspose2d

class EDSR(nn.Module):
    def __init__(self, args):
        super(EDSR, self).__init__()
        n_resblocks = args.n_res_blocks
        n_feats = args.n_feats
        scale = args.scale
        conv = common.default_conv


        kernel_size = 3
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(1)
        self.add_mean = common.MeanShift(1, sign=1)

        deconv = None
        if args.deconv:
            deconv = NormConvTranspose2d if args.norm else nn.ConvTranspose2d
            name = 'NormDedonv' if args.norm else 'Deconv'

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        if deconv is not None:
            upsampler = common.DeconvUpsampler(deconv, scale, n_feats, act=False)
        else:
            upsampler = common.Upsampler(conv, scale, n_feats, act=False)

        m_tail = [
            upsampler,
            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
