''' Pyramid-Context Encoder Networks: PEN-Net
'''
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# #################################################################################
# ########################  Contextual Attention  #################################
# #################################################################################
'''
implementation of attention module
most codes are borrowed from:
1. https://github.com/WonwoongCho/Generative-Inpainting-pytorch/pull/5/commits/9c16537cd123b74453a28cd4e25d3db0505e5881
2. https://github.com/DAA233/generative-inpainting-pytorch/blob/master/model/networks.py
'''

class AtnConv(nn.Module):
  def __init__(self, input_channels=128, output_channels=64, groups=4, ksize=3, stride=1, rate=2, softmax_scale=10., fuse=True, rates=[1,2,4,8]):
    super(AtnConv, self).__init__()
    self.ksize = ksize
    self.stride = stride
    self.rate = rate 
    self.softmax_scale = softmax_scale
    self.groups = groups
    self.fuse = fuse
    if self.fuse:
      for i in range(groups):
        self.__setattr__('conv{}'.format(str(i).zfill(2)), nn.Sequential(
          nn.Conv2d(input_channels, output_channels//groups, kernel_size=3, dilation=rates[i], padding=rates[i]),
          nn.ReLU(inplace=True))
        )
    
  def forward(self, x1, x2, mask=None):
    """ Attention Transfer Network (ATN) is first proposed in
        Learning Pyramid Context-Encoder Networks for High-Quality Image Inpainting. Yanhong Zeng et al. In CVPR 2019.
      inspired by 
        Generative Image Inpainting with Contextual Attention, Yu et al. In CVPR 2018. 
    Args:
        x1: low-level feature maps with larger resolution.
        x2: high-level feature maps with smaller resolution. 
        mask: Input mask, 1 indicates holes. 
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from b.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.
    Returns:
        torch.Tensor, reconstructed feature map. 
    """
    # get shapes
    x1s = list(x1.size())
    x2s = list(x2.size())

    # extract patches from low-level feature maps x1 with stride and rate
    kernel = 2*self.rate
    raw_w = extract_patches(x1, kernel=kernel, stride=self.rate*self.stride)
    raw_w = raw_w.contiguous().view(x1s[0], -1, x1s[1], kernel, kernel) # B*HW*C*K*K 
    # split tensors by batch dimension; tuple is returned
    raw_w_groups = torch.split(raw_w, 1, dim=0) 

    # split high-level feature maps x2 for matching 
    f_groups = torch.split(x2, 1, dim=0) 
    # extract patches from x2 as weights of filter
    w = extract_patches(x2, kernel=self.ksize, stride=self.stride)
    w = w.contiguous().view(x2s[0], -1, x2s[1], self.ksize, self.ksize) # B*HW*C*K*K
    w_groups = torch.split(w, 1, dim=0) 

    # process mask
    if mask is not None:
      mask = F.interpolate(mask, size=x2s[2:4], mode='bilinear', align_corners=True)
    else:
      mask = torch.zeros([1, 1, x2s[2], x2s[3]])
      if torch.cuda.is_available():
        mask = mask.cuda()
    # extract patches from masks to mask out hole-patches for matching 
    m = extract_patches(mask, kernel=self.ksize, stride=self.stride)
    m = m.contiguous().view(x2s[0], -1, 1, self.ksize, self.ksize)  # B*HW*1*K*K
    m = m.mean([2,3,4]).unsqueeze(-1).unsqueeze(-1)
    mm = m.eq(0.).float() # (B, HW, 1, 1)       
    mm_groups = torch.split(mm, 1, dim=0)

    y = []
    scale = self.softmax_scale
    padding = 0 if self.ksize==1 else 1
    for xi, wi, raw_wi, mi in zip(f_groups, w_groups, raw_w_groups, mm_groups):
      '''
      O => output channel as a conv filter
      I => input channel as a conv filter
      xi : separated tensor along batch dimension of front; 
      wi : separated patch tensor along batch dimension of back; 
      raw_wi : separated tensor along batch dimension of back; 
      '''
      # matching based on cosine-similarity
      wi = wi[0]
      escape_NaN = torch.FloatTensor([1e-4])
      if torch.cuda.is_available():
        escape_NaN = escape_NaN.cuda()
      # normalize 
      wi_normed = wi / torch.max(torch.sqrt((wi*wi).sum([1,2,3],keepdim=True)), escape_NaN)
      yi = F.conv2d(xi, wi_normed, stride=1, padding=padding)
      yi = yi.contiguous().view(1, x2s[2]//self.stride*x2s[3]//self.stride, x2s[2], x2s[3]) 

      # apply softmax to obtain 
      yi = yi * mi 
      yi = F.softmax(yi*scale, dim=1)
      yi = yi * mi
      yi = yi.clamp(min=1e-8)

      # attending 
      wi_center = raw_wi[0]
      yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.
      y.append(yi)
    y = torch.cat(y, dim=0)
    y.contiguous().view(x1s)
    # adjust after filling 
    if self.fuse:
      tmp = []
      for i in range(self.groups):
        tmp.append(self.__getattr__('conv{}'.format(str(i).zfill(2)))(y))
      y = torch.cat(tmp, dim=1)
    return y


# extract patches
def extract_patches(x, kernel=3, stride=1):
  if kernel != 1:
    x = nn.ZeroPad2d(1)(x)
  x = x.permute(0, 2, 3, 1)
  all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
  return all_patches


# codes borrowed from https://github.com/DAA233/generative-inpainting-pytorch/blob/master/model/networks.py
def test_contextual_attention(args):
  """Test contextual attention layer with 3-channel image input
  (instead of n-channel feature).
  """
  rate = 2
  stride = 1
  grid = rate*stride
  import cv2
  import numpy as np
  import matplotlib.pyplot as plt

  b = cv2.imread(args[1])
  b = cv2.resize(b, (b.shape[0]//2, b.shape[1]//2))
  print(args[1])
  h, w, c = b.shape
  b = b[:h//grid*grid, :w//grid*grid, :]
  b = np.transpose(b,[2,0,1])
  b = np.expand_dims(b, 0)
  print('Size of imageA: {}'.format(b.shape))

  f = cv2.imread(args[2])
  h, w, _ = f.shape
  f = f[:h//grid*grid, :w//grid*grid, :]
  f = np.transpose(f,[2,0,1])
  f = np.expand_dims(f, 0)
  print('Size of imageB: {}'.format(f.shape))

  bt = torch.Tensor(b)
  ft = torch.Tensor(f)
  atnconv = AtnConv(stride=stride, fuse=False)
  yt = atnconv(ft, bt)
  y = yt.cpu().data.numpy().transpose([0,2,3,1])
  outImg = np.clip(y[0],0,255).astype(np.uint8)
  plt.imshow(outImg)
  plt.show()
  print(outImg.shape)
  cv2.imwrite('output.jpg', outImg)


if __name__ == '__main__':
  import sys
  test_contextual_attention(sys.argv)