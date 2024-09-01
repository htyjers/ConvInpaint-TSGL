import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
from spade import SPADE
from torch.nn import init

import matplotlib.pyplot as plt
from saicinpainting.training.modules.base import get_activation, BaseDiscriminator
from saicinpainting.training.modules.spatial_transform import LearnableSpatialTransformWrapper
from saicinpainting.training.modules.squeeze_excitation import SELayer
from saicinpainting.utils import get_shape


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
                                              self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
                                              self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode,
                              align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        # print("------------")
        x_l, x_g = self.ffc(x)
        # print(x_l.shape)
        x_l = self.act_l(self.bn_l(x_l))
        # print(x_l.shape)
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class SPDNormResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc_low, semantic_nc_high, norm_G='spectralspadesyncbatch3x3'):
        super(SPDNormResnetBlock, self).__init__()
        # Attributes
        self.learned_shortcut = True
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        self.conv_s = nn.Conv2d(fin, fout, kernel_size=3, padding=1)

        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc_high)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc_high)
        self.norm_s = SPADE('spadeposition3x3', fin, semantic_nc_low)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg1, seg2):
        # Residual low frequency--seg1
        low = self.conv_s(self.actvn(self.norm_s(x, seg1)))
        # High frquency--seg2
        high = self.conv_0(self.actvn(self.norm_0(x, seg2)))
        high = self.conv_1(self.actvn(self.norm_1(high, seg2)))

        out = high + low + x
        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class FFCResNetGenerator(BaseNetwork):
    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={'ratio_gin': 0, 'ratio_gout': 0, 'enable_lfu': False},
                 downsample_conv_kwargs={'ratio_gin': 0, 'ratio_gout': 0, 'enable_lfu': False},
                 resnet_conv_kwargs={'ratio_gin': 0.75, 'ratio_gout': 0.75, 'enable_lfu': False},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act='sigmoid', max_features=256, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        #####
        n_downsampling = 5

        model_high = [nn.ReflectionPad2d(3),
                      FFC_BN_ACT(3, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                                 activation_layer=activation_layer, **init_conv_kwargs)]

        self.model_high = nn.Sequential(*model_high)
        i = 0
        mult = 2 ** i
        if i == n_downsampling - 1:
            cur_conv_kwargs = dict(downsample_conv_kwargs)
            cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
        else:
            cur_conv_kwargs = downsample_conv_kwargs
        model_high_0 = [FFC_BN_ACT(min(max_features, ngf * mult),
                                   min(max_features, ngf * mult * 2),
                                   kernel_size=3, stride=2, padding=1,
                                   norm_layer=norm_layer,
                                   activation_layer=activation_layer,
                                   **cur_conv_kwargs)]
        self.model_high_0 = nn.Sequential(*model_high_0)

        i = 1
        mult = 2 ** i
        if i == n_downsampling - 1:
            cur_conv_kwargs = dict(downsample_conv_kwargs)
            cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
        else:
            cur_conv_kwargs = downsample_conv_kwargs
        model_high_1 = [FFC_BN_ACT(min(max_features, ngf * mult),
                                   min(max_features, ngf * mult * 2),
                                   kernel_size=3, stride=2, padding=1,
                                   norm_layer=norm_layer,
                                   activation_layer=activation_layer,
                                   **cur_conv_kwargs)]
        self.model_high_1 = nn.Sequential(*model_high_1)

        i = 2
        mult = 2 ** i
        if i == n_downsampling - 1:
            cur_conv_kwargs = dict(downsample_conv_kwargs)
            cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
        else:
            cur_conv_kwargs = downsample_conv_kwargs
        model_high_2 = [FFC_BN_ACT(min(max_features, ngf * mult),
                                   min(max_features, ngf * mult * 2),
                                   kernel_size=3, stride=2, padding=1,
                                   norm_layer=norm_layer,
                                   activation_layer=activation_layer,
                                   **cur_conv_kwargs)]
        self.model_high_2 = nn.Sequential(*model_high_2)

        i = 3
        mult = 2 ** i
        if i == n_downsampling - 1:
            cur_conv_kwargs = dict(downsample_conv_kwargs)
            cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
        else:
            cur_conv_kwargs = downsample_conv_kwargs
        model_high_3 = [FFC_BN_ACT(min(max_features, ngf * mult),
                                   min(max_features, ngf * mult * 2),
                                   kernel_size=3, stride=2, padding=1,
                                   norm_layer=norm_layer,
                                   activation_layer=activation_layer,
                                   **cur_conv_kwargs)]
        self.model_high_3 = nn.Sequential(*model_high_3)

        i = 4
        mult = 2 ** i
        if i == n_downsampling - 1:
            cur_conv_kwargs = dict(downsample_conv_kwargs)
            cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
        else:
            cur_conv_kwargs = downsample_conv_kwargs
        model_high_4 = [FFC_BN_ACT(min(max_features, ngf * mult),
                                   min(max_features, ngf * mult * 2),
                                   kernel_size=3, stride=2, padding=1,
                                   norm_layer=norm_layer,
                                   activation_layer=activation_layer,
                                   **cur_conv_kwargs)]
        self.model_high_4 = nn.Sequential(*model_high_4)

        #####low
        model_low = [nn.ReflectionPad2d(3),
                     FFC_BN_ACT(input_nc, min(max_features, ngf * (2 ** (n_downsampling))), kernel_size=7, padding=0,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer, **init_conv_kwargs)]
        self.model_low = nn.Sequential(*model_low)

        i = 0
        if i == n_downsampling - 1:
            cur_conv_kwargs = dict(downsample_conv_kwargs)
            cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
        else:
            cur_conv_kwargs = downsample_conv_kwargs
        model_low_0 = [FFC_BN_ACT(min(max_features, ngf * (2 ** (n_downsampling))),
                                  min(max_features, ngf * (2 ** (n_downsampling))),
                                  kernel_size=3, stride=2, padding=1,
                                  norm_layer=norm_layer,
                                  activation_layer=activation_layer,
                                  **cur_conv_kwargs)]
        self.model_low_0 = nn.Sequential(*model_low_0)

        i = 1
        if i == n_downsampling - 1:
            cur_conv_kwargs = dict(downsample_conv_kwargs)
            cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
        else:
            cur_conv_kwargs = downsample_conv_kwargs
        model_low_1 = [FFC_BN_ACT(min(max_features, ngf * (2 ** (n_downsampling))),
                                  min(max_features, ngf * (2 ** (n_downsampling))),
                                  kernel_size=3, stride=2, padding=1,
                                  norm_layer=norm_layer,
                                  activation_layer=activation_layer,
                                  **cur_conv_kwargs)]
        self.model_low_1 = nn.Sequential(*model_low_1)

        i = 2
        if i == n_downsampling - 1:
            cur_conv_kwargs = dict(downsample_conv_kwargs)
            cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
        else:
            cur_conv_kwargs = downsample_conv_kwargs
        model_low_2 = [FFC_BN_ACT(min(max_features, ngf * (2 ** (n_downsampling))),
                                  min(max_features, ngf * (2 ** (n_downsampling))),
                                  kernel_size=3, stride=2, padding=1,
                                  norm_layer=norm_layer,
                                  activation_layer=activation_layer,
                                  **cur_conv_kwargs)]
        self.model_low_2 = nn.Sequential(*model_low_2)

        i = 3
        if i == n_downsampling - 1:
            cur_conv_kwargs = dict(downsample_conv_kwargs)
            cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
        else:
            cur_conv_kwargs = downsample_conv_kwargs
        model_low_3 = [FFC_BN_ACT(min(max_features, ngf * (2 ** (n_downsampling))),
                                  min(max_features, ngf * (2 ** (n_downsampling))),
                                  kernel_size=3, stride=2, padding=1,
                                  norm_layer=norm_layer,
                                  activation_layer=activation_layer,
                                  **cur_conv_kwargs)]
        self.model_low_3 = nn.Sequential(*model_low_3)

        i = 4
        if i == n_downsampling - 1:
            cur_conv_kwargs = dict(downsample_conv_kwargs)
            cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
        else:
            cur_conv_kwargs = downsample_conv_kwargs
        model_low_4 = [FFC_BN_ACT(min(max_features, ngf * (2 ** (n_downsampling))),
                                  min(max_features, ngf * (2 ** (n_downsampling))),
                                  kernel_size=3, stride=2, padding=1,
                                  norm_layer=norm_layer,
                                  activation_layer=activation_layer,
                                  **cur_conv_kwargs)]
        self.model_low_4 = nn.Sequential(*model_low_4)

        ############## Fusion branch #############
        self.h2l_0_l = SPDNormResnetBlock(max_features, max_features, max_features, ngf * 2)
        self.h2l_1_l = SPDNormResnetBlock(max_features, max_features, max_features, ngf * 4)
        self.h2l_2_l = SPDNormResnetBlock(max_features, max_features, max_features, max_features)
        self.h2l_3_l = SPDNormResnetBlock(max_features, max_features, max_features, max_features)
        self.h2l_4_l = SPDNormResnetBlock(max_features//4,max_features//4,max_features//4,max_features//4)
        self.h2l_4_g = SPDNormResnetBlock(max_features*3//4,max_features*3//4,max_features*3//4,max_features*3//4)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        model = []
        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type,
                                          activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # if add_out_act:
        model.append(get_activation('tanh'))
        self.model = nn.Sequential(*model)

        ###high  up####

        up_high = []
        for i in range(3):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type,
                                          activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            up_high += [cur_resblock]

        up_high += [ConcatTupleLayer()]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            up_high += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                           min(max_features, int(ngf * mult / 2)),
                                           kernel_size=3, stride=2, padding=1, output_padding=1),
                        up_norm_layer(min(max_features, int(ngf * mult / 2))),
                        up_activation]
        if out_ffc:
            up_high += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                       norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        up_high += [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, 1, kernel_size=7, padding=0)]
        if add_out_act:
            up_high.append(get_activation('sigmoid' if add_out_act is True else add_out_act))
        self.up_high = nn.Sequential(*up_high)

        self.up_high = nn.Sequential(*up_high)

        self.init_weights()

    def forward(self, image, high, gray, mask):
        input_high_l, input_high_g = self.model_high(torch.cat((high, gray, mask), dim=1))
        input_low_l, input_low_g = self.model_low(torch.cat((image, mask), dim=1))

        ###################
        input_high_0_l, input_high_0_g = self.model_high_0((input_high_l, input_high_g))
        input_low_0_l, input_low_0_g = self.model_low_0((input_low_l, input_low_g))
        input_low_high_0_l = input_low_l - self.up(input_low_0_l)
        input_low_0_l = self.h2l_0_l(input_low_0_l, input_low_high_0_l, input_high_0_l)

        ###################
        input_high_1_l, input_high_1_g = self.model_high_1((input_high_0_l, input_high_0_g))
        input_low_1_l, input_low_1_g = self.model_low_1((input_low_0_l, input_low_0_g))
        input_low_high_1_l = input_low_0_l - self.up(input_low_1_l)
        input_low_1_l = self.h2l_1_l(input_low_1_l, input_low_high_1_l, input_high_1_l)

        ###################
        input_high_2_l, input_high_2_g = self.model_high_2((input_high_1_l, input_high_1_g))
        input_low_2_l, input_low_2_g = self.model_low_2((input_low_1_l, input_low_1_g))
        input_low_high_2_l = input_low_1_l - self.up(input_low_2_l)
        input_low_2_l = self.h2l_2_l(input_low_2_l, input_low_high_2_l, input_high_2_l)

        ###################
        input_high_3_l, input_high_3_g = self.model_high_3((input_high_2_l, input_high_2_g))
        input_low_3_l, input_low_3_g = self.model_low_3((input_low_2_l, input_low_2_g))
        input_low_high_3_l = input_low_2_l - self.up(input_low_3_l)
        input_low_3_l = self.h2l_3_l(input_low_3_l, input_low_high_3_l, input_high_3_l)

        ###################
        input_high_4_l, input_high_4_g = self.model_high_4((input_high_3_l, input_high_3_g))
        input_low_4_l, input_low_4_g = self.model_low_4((input_low_3_l, input_low_3_g))
        input_low_high_4_l = input_low_3_l - self.up(torch.cat([input_low_4_l,input_low_4_g], dim=1))
        input_low_high_4_ls = torch.split(input_low_high_4_l, [256 // 4, 256 * 3 // 4], dim=1)
        input_low_4_l = self.h2l_4_l(input_low_4_l, input_low_high_4_ls[0], input_high_4_l)
        input_low_4_g = self.h2l_4_g(input_low_4_g, input_low_high_4_ls[1], input_high_4_g)
        #

        return self.model((input_low_4_l, input_low_4_g)), self.up_high((input_high_4_l, input_high_4_g))


# Defines the PatchGAN discriminator with the specified arguments.
class Discriminator(BaseDiscriminator):
    def __init__(self, input_nc=3, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d, ):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]

