import os
import os.path as osp
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from PIL import Image
from spade import SPADE
import torch.nn as nn
from torch.nn import init
from pconv import PConvBNActiv
from trans import BasicLayer, PatchMerging, PatchUpsampling, token2feature, feature2token, Conv2dLayerPartial, DecTranBlock
from spectral_norm import use_spectral_norm

from pcconv_cross import PCconv
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


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = self.relu(out)
        
        return out

class Feature2High(nn.Module):

    def __init__(self, inplanes, planes):
        super(Feature2High, self).__init__()

        self.structure_resolver = Bottleneck(inplanes, planes)
        self.out_layer = nn.Sequential(
            nn.Conv2d(inplanes, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, structure_feature):

        x = self.structure_resolver(structure_feature)
        structure = self.out_layer(x)
        return structure
      

      
class SPDNormResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc_low, semantic_nc_high, norm_G = 'spectralspadesyncbatch3x3'):#spadeposition3x3
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

        out = high + low
        return out


    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class UnetSkipConnectionDBlock(nn.Module):
    def __init__(self, inner_nc, outer_nc, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionDBlock, self).__init__()
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)
        upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                    kernel_size=4, stride=2,
                                    padding=1)
        up = [uprelu, upconv, upnorm]

        if outermost:
            up = [uprelu, upconv, nn.Tanh()]
            model = up
        elif innermost:
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            up = [uprelu, upconv, upnorm]
            model = up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class Generator(BaseNetwork):

    def __init__(self,activation='lrelu'):
        super(Generator, self).__init__()
        
        cnum = 64
        tdim = 180
        #################    #################    #################
        ##   Encoder   ##    ##   Encoder   ##    ##   Encoder   ##
        #################    #################    #################

        ############## high frequency branch -- CNN ##############  0 is masked, 1 is known
        # mask + image + high = 5 channel 
        self.en_high_1 = PConvBNActiv(3, cnum, bn=False, sample='down-7') #128
        self.en_high_2 = PConvBNActiv(cnum, cnum*2, sample='down-5') #64
        self.en_high_3 = PConvBNActiv(cnum*2, cnum*4, sample='down-5') #32
        self.en_high_4 = PConvBNActiv(cnum*4, cnum*8, sample='down-3') #16
        self.en_high_5 = PConvBNActiv(cnum*8, cnum*16, sample='down-3') #8

        ############## low frequency branch -- Transformer ##############  0 is masked，1 is known
        # mask + image + low = 7 channel
        #1. Head
        self.en_low_1_f = Conv2dLayerPartial(in_channels=7, out_channels=tdim, kernel_size=3, activation=activation)
        self.en_low_1_s = Conv2dLayerPartial(in_channels=tdim, out_channels=tdim, kernel_size=3, down=2, activation=activation)
        #128
        
        #2. Body
        # from 128 -> 64 -> 32 -> 16 -> 8
        depths = [2, 2, 6, 2] # num of layer
        ratios = [1/2, 1/2, 1/2, 1/2]
        #tnum = [cnum, cnum*2, cnum*4, cnum*8, cnum*16] # channel
        tnum = [tdim,tdim,tdim,tdim,tdim] # channel
        num_heads = 4 
        window_sizes = [8, 8, 8, 8]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        merge2 = PatchMerging(tnum[0], tnum[1], down=int(1/ratios[0]))
        self.en_low_2 = BasicLayer(dim=tnum[1], input_resolution=[64, 64], depth=depths[0], num_heads=num_heads,
                           window_size=window_sizes[0], drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                           downsample=merge2) #64

        merge3 = PatchMerging(tnum[1], tnum[2], down=int(1/ratios[1]))
        self.en_low_3 = BasicLayer(dim=tnum[2], input_resolution=[32, 32], depth=depths[1], num_heads=num_heads,
                           window_size=window_sizes[1], drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                           downsample=merge3) #32

        merge4 = PatchMerging(tnum[2], tnum[3], down=int(1/ratios[2]))
        self.en_low_4 = BasicLayer(dim=tnum[3], input_resolution=[16, 16], depth=depths[2], num_heads=num_heads,
                           window_size=window_sizes[2], drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                           downsample=merge4) #16

        merge5 = PatchMerging(tnum[3], tnum[4], down=int(1/ratios[3]))
        self.en_low_5 = BasicLayer(dim=tnum[4], input_resolution=[8, 8], depth=depths[3], num_heads=num_heads,
                           window_size=window_sizes[3], drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                           downsample=merge5) #8

        ############## Fusion branch ##############
        self.en_fusion_1 = nn.Sequential(
            nn.Conv2d(tdim, cnum, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

        self.en_fusion_2 = nn.Sequential(
            nn.Conv2d(tdim, cnum * 2, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

        self.en_fusion_3 = nn.Sequential(
            nn.Conv2d(tdim, cnum * 4, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

        self.en_fusion_4 = nn.Sequential(
            nn.Conv2d(tdim, cnum * 8, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

        self.en_fusion_5 = nn.Sequential(
            nn.Conv2d(tdim, cnum * 8, kernel_size=1, padding = 0, bias=True),
            nn.ReLU(inplace=True)
        )

        ############## Guide ##############
        # High -> Low
        
        self.h2l_1 = SPDNormResnetBlock(tdim,tdim,tdim,cnum)
        self.h2l_2 = SPDNormResnetBlock(tdim,tdim,tdim,cnum*2)
        self.h2l_3 = SPDNormResnetBlock(tdim,tdim,tdim,cnum*4)
        self.h2l_4 = SPDNormResnetBlock(tdim,tdim,tdim,cnum*8)
        self.h2l_5 = SPDNormResnetBlock(tdim,tdim,tdim,cnum*16)

        #################    #################    #################
        ##   Decoder   ##    ##   Decoder   ##    ##   Decoder   ##
        #################    #################    #################
        
        ############## Fusion branch ##############
        self.en_fusion_6 = nn.Sequential(
            nn.Conv2d(cnum * (8 + 8 + 4 + 2 + 1), cnum * (8), kernel_size=3, stride=2,padding = 1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        self.pc_block = PCconv()
        
        norm_layer = nn.BatchNorm2d
        use_dropout=False
        self.Decoder_1 = UnetSkipConnectionDBlock(cnum * 8, cnum * 8, norm_layer=norm_layer, use_dropout=use_dropout,innermost=True)
        self.Decoder_2 = UnetSkipConnectionDBlock(cnum * 16, cnum * 8, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Decoder_3 = UnetSkipConnectionDBlock(cnum * 16, cnum * 4, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Decoder_4 = UnetSkipConnectionDBlock(cnum * 8, cnum * 2, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Decoder_5 = UnetSkipConnectionDBlock(cnum * 4, cnum, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Decoder_6 = UnetSkipConnectionDBlock(cnum * 2, 3, norm_layer=norm_layer, use_dropout=use_dropout, outermost=True)
        

        ############## High frequency branch -- CNN ##############  
        #U-net without skip connect
        self.en_high_6 = PConvBNActiv(cnum*16, cnum*16, sample='down-3')#4 
        self.en_high_7 = PConvBNActiv(cnum*16, cnum*16, sample='down-3')#2

        self.de_high_7 = PConvBNActiv(cnum*16, cnum*16, activ='leaky')#4
        self.de_high_6 = PConvBNActiv(cnum*16, cnum*16, activ='leaky')#8
        self.de_high_5 = PConvBNActiv(cnum*16, cnum*16, activ='leaky')#16
        self.de_high_4 = PConvBNActiv(cnum*16, cnum*8, activ='leaky')#32
        self.de_high_3 = PConvBNActiv(cnum*8, cnum*4, activ='leaky')#64
        self.de_high_2 = PConvBNActiv(cnum*4, cnum*2, activ='leaky')#128
        self.de_high_1 = PConvBNActiv(cnum*2, cnum, activ='leaky')#256
        self.de_high_0 = Feature2High(cnum, cnum)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.init_weights()
        
    def forward(self, image, high, low, gray, mask):
        
        # 1 is known and 0 is masked
        #################    #################    #################
        ##   Encoder   ##    ##   Encoder   ##    ##   Encoder   ##
        #################    #################    #################
        
        ############## Multi-frenquency branch ##############
        ## Layer 0
        B,_,_,_ = image.shape
        layer_h0 = torch.cat((high, gray, mask), dim=1)
        layer_h0_mask =  torch.cat((mask, mask, mask), dim=1)
        layer_l0 = torch.cat((image, low, mask), dim=1)
        layer_l0_mask = mask
        
        ## Layer 1
        layer_h1,layer_h1_mask = self.en_high_1(layer_h0,layer_h0_mask)
        layer_l1_t1,layer_l1_mask = self.en_low_1_f(layer_l0, layer_l0_mask)
        layer_l1_t2,layer_l1_mask = self.en_low_1_s(layer_l1_t1, layer_l1_mask)
        
        layer_l1_t3 = F.interpolate(layer_l1_t1 - self.up(layer_l1_t2), size=layer_h1.size()[2:4], mode='bilinear', align_corners=True)
        layer_l1 = self.h2l_1(layer_l1_t2, layer_l1_t3, layer_h1)

        ## Layer 2
        layer_h2,layer_h2_mask = self.en_high_2(layer_h1,layer_h1_mask)
        layer_l2_t1, layer_l2_size, layer_l2_mask = self.en_low_2(feature2token(layer_l1), layer_l1.size()[-2:], feature2token(layer_l1_mask))
        layer_l2_t2 = token2feature(layer_l2_t1, layer_l2_size).contiguous()
        
        layer_l2_t3 = F.interpolate(layer_l1 - self.up(layer_l2_t2), size=layer_h2.size()[2:4], mode='bilinear', align_corners=True)
        layer_l2 = self.h2l_2(layer_l2_t2, layer_l2_t3, layer_h2)

        ## Layer 3
        layer_h3,layer_h3_mask = self.en_high_3(layer_h2,layer_h2_mask)
        layer_l3_t1, layer_l3_size, layer_l3_mask = self.en_low_3(feature2token(layer_l2), layer_l2_size, layer_l2_mask)
        layer_l3_t2 = token2feature(layer_l3_t1, layer_l3_size).contiguous()
        
        layer_l3_t3 = F.interpolate(layer_l2 - self.up(layer_l3_t2), size=layer_h3.size()[2:4], mode='bilinear', align_corners=True)
        layer_l3 = self.h2l_3(layer_l3_t2, layer_l3_t3, layer_h3)
        

        ## Layer 4
        layer_h4,layer_h4_mask = self.en_high_4(layer_h3,layer_h3_mask)
        layer_l4_t1, layer_l4_size, layer_l4_mask = self.en_low_4(feature2token(layer_l3), layer_l3_size, layer_l3_mask)
        layer_l4_t2 = token2feature(layer_l4_t1, layer_l4_size).contiguous()
        
        layer_l4_t3 = F.interpolate(layer_l3 - self.up(layer_l4_t2), size=layer_h4.size()[2:4], mode='bilinear', align_corners=True)
        layer_l4 = self.h2l_4(layer_l4_t2, layer_l4_t3, layer_h4)
        

        ## Layer 5
        layer_h5,layer_h5_mask = self.en_high_5(layer_h4,layer_h4_mask)
        layer_l5_t1, layer_l5_size, layer_l5_mask = self.en_low_5(feature2token(layer_l4), layer_l4_size, None)
        layer_l5_t2 = token2feature(layer_l5_t1, layer_l5_size).contiguous()
        
        layer_l5_t3 = F.interpolate(layer_l4 - self.up(layer_l5_t2), size=layer_h5.size()[2:4], mode='bilinear', align_corners=True)
        layer_l5 = self.h2l_5(layer_l5_t2, layer_l5_t3, layer_h5)
        
        ############## Fusion branch ##############
        # layer_f1 = self.en_fusion_1(torch.cat((layer_h1, layer_l1), dim=1)).reshape(B,-1,128,128)
        # layer_f2 = self.en_fusion_2(torch.cat((layer_h2, layer_l2), dim=1)).reshape(B,-1,64,64)
        # layer_f3 = self.en_fusion_3(torch.cat((layer_h3, layer_l3), dim=1)).reshape(B,-1,32,32)
        # layer_f4 = self.en_fusion_4(torch.cat((layer_h4, layer_l4), dim=1)).reshape(B,-1,16,16)
        # layer_f5 = self.en_fusion_5(torch.cat((layer_h5, layer_l5), dim=1)).reshape(B,-1,8,8) #

        layer_f1 = self.en_fusion_1(layer_l1).reshape(B,-1,128,128)
        layer_f2 = self.en_fusion_2(layer_l2).reshape(B,-1,64,64)
        layer_f3 = self.en_fusion_3(layer_l3).reshape(B,-1,32,32)
        layer_f4 = self.en_fusion_4(layer_l4).reshape(B,-1,16,16)
        layer_f5 = self.en_fusion_5(layer_l5).reshape(B,-1,8,8) #
        
        #################    #################    #################
        ##   Decoder   ##    ##   Decoder   ##    ##   Decoder   ##
        #################    #################    #################

        ############## Fusion branch ##############  
        ## Layer 6
        layer_f1_d = F.interpolate(layer_f1, scale_factor=1/16, mode='bilinear', align_corners=True)
        layer_f2_d = F.interpolate(layer_f2, scale_factor=1/8, mode='bilinear', align_corners=True)
        layer_f3_d = F.interpolate(layer_f3, scale_factor=1/4, mode='bilinear', align_corners=True)
        layer_f4_d = F.interpolate(layer_f4, scale_factor=1/2, mode='bilinear', align_corners=True)
        layer_f6 = self.en_fusion_6(torch.cat((layer_f1_d, layer_f2_d, layer_f3_d, layer_f4_d, layer_f5), dim=1))
        
        #########################################################
        x_out = self.pc_block([layer_f1, layer_f2, layer_f3, layer_f4, layer_f5, layer_f6],1-mask)
        y_1 = self.Decoder_1(x_out[5])
        y_2 = self.Decoder_2(torch.cat([y_1, x_out[4]], 1))
        y_3 = self.Decoder_3(torch.cat([y_2, x_out[3]], 1))
        y_4 = self.Decoder_4(torch.cat([y_3, x_out[2]], 1))
        y_5 = self.Decoder_5(torch.cat([y_4, x_out[1]], 1))
        output = self.Decoder_6(torch.cat([y_5, x_out[0]], 1))

        #########################################################
        
        ############## High frequency branch -- CNN #############
        layer_h6,layer_h6_mask = self.en_high_6(layer_h5,layer_h5_mask)
        layer_h7,layer_h7_mask = self.en_high_7(layer_h6,layer_h6_mask)
        de_high, de_high_masks = layer_h7,layer_h7_mask
        for _ in range(7, 0, -1):
            de_conv = 'de_high_{:d}'.format(_)
            de_high = F.interpolate(de_high, scale_factor=2, mode='bilinear')
            de_high_masks = F.interpolate(de_high_masks, scale_factor=2, mode='nearest')
            de_high, de_high_masks = getattr(self, de_conv)(de_high, de_high_masks)
        image_high = self.de_high_0(de_high)

        return output, image_high


class Discriminator(BaseNetwork):
  def __init__(self, in_channels, use_sigmoid=False, use_sn=True, init_weights=True):
    super(Discriminator, self).__init__()
    self.use_sigmoid = use_sigmoid
    cnum = 64
    self.encoder = nn.Sequential(
      use_spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=cnum,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum, out_channels=cnum*2,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),
      
      use_spectral_norm(nn.Conv2d(in_channels=cnum*2, out_channels=cnum*4,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum*4, out_channels=cnum*8,
        kernel_size=5, stride=1, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),
    )

    self.classifier = nn.Conv2d(in_channels=cnum*8, out_channels=1, kernel_size=5, stride=1, padding=1)
    if init_weights:
      self.init_weights()


  def forward(self, x):
    x = self.encoder(x)
    label_x = self.classifier(x)
    if self.use_sigmoid:
      label_x = torch.sigmoid(label_x)
    return label_x


def requires_grad(model, flag=True):
    for p in model.parameters():
      p.requires_grad = flag


      
class UnetSkipConnectionEBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionEBlock, self).__init__()
        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)

        downrelu = nn.LeakyReLU(0.2, True)

        downnorm = norm_layer(inner_nc, affine=True)
        if outermost:
            down = [downconv]
            model = down
        elif innermost:
            down = [downrelu, downconv]
            model = down
        else:
            down = [downrelu, downconv, downnorm]
            if use_dropout:
                model = down + [nn.Dropout(0.5)]
            else:
                model = down
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=False),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=False),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
