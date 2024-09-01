""" Trainer for meta-train phase. """
import json
import os
import os.path as osp
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image
import math

from dataset_loader import DatasetLoader as Dataset
from dataset_loader1 import DatasetLoader as Dataset1
from medfe import Generator,Discriminator,requires_grad
from itertools import cycle
import random
import warnings
warnings.filterwarnings("ignore")
class Trainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = '/data/256/inpaint_log'
        meta_base_dir = osp.join(log_base_dir, args.file_name)
        self.save_path = meta_base_dir
        if os.path.exists(self.save_path):
            pass
        else:
            os.makedirs(self.save_path)
        self.args = args

        self.trainset = Dataset('train', self.args)
        self.trainset1 = Dataset1('train', self.args)
        self.netG = Generator().to(self.args.device)

        
        gan_type = 'hinge'
        self.netD = Discriminator(in_channels=3, use_sigmoid = gan_type != 'hinge').to(self.args.device)
        
        self.adversarial_loss = AdversarialLoss(type = gan_type)
        self.l1loss = nn.L1Loss()
        
        self.optimizer_g = torch.optim.Adam(self.netG.parameters(), lr = 5e-4, betas = (0.5, 0.99))
        self.optimizer_d = torch.optim.Adam(self.netD.parameters(), lr = 1e-4, betas = (0.5, 0.99))
         
        self.lr_scheduler_g = torch.optim.lr_scheduler.StepLR(self.optimizer_g, 3, gamma=0.1)#70
        self.lr_scheduler_d = torch.optim.lr_scheduler.StepLR(self.optimizer_d, 3, gamma=0.1)
        
        self.netG = torch.nn.DataParallel(self.netG)
        self.netD = torch.nn.DataParallel(self.netD)

    def train(self):
        self.netG.train()
        self.netD.train()
        print(self.netG)
        print(self.netD)
        self.train_loader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.train_loader1 = DataLoader(self.trainset1, batch_size=self.args.batch_size, shuffle=True, num_workers=4, drop_last=True)

        for epoch in range(1, self.args.max_epoch + 1):
            i = 0
            mask_iterator = iter(self.train_loader1)
            for (image, high, low, gray) in self.train_loader:
                image = image.to(self.args.device)
                high = high.to(self.args.device)
                low = low.to(self.args.device)
                gray = gray.to(self.args.device)
                try:
                    mask = next(mask_iterator).to(self.args.device)
                except StopIteration:
                    mask_iterator = iter(self.train_loader1)
                    mask = next(mask_iterator).to(self.args.device)
                B,C,H,W = image.size()
                output, image_high, image_low = self.netG(image * mask, high * mask, low * mask, gray * mask, mask)
                
                com_output = image * mask + output * (1 - mask)
                
                requires_grad(self.netD, True)
                requires_grad(self.netG, False)
                # Discriminator loss
                dis_real_feat = self.netD(image)                   
                dis_fake_feat = self.netD(com_output.detach())     
                dis_real_loss = self.adversarial_loss(dis_real_feat, True, True)
                dis_fake_loss = self.adversarial_loss(dis_fake_feat, False, True)
                dis_loss = (dis_real_loss + dis_fake_loss) / 2
                self.optimizer_d.zero_grad()
                dis_loss.backward()
                self.optimizer_d.step()
                
                requires_grad(self.netG, True)
                requires_grad(self.netD, False)
                # Generator Loss
                gen_fake_feat = self.netD(com_output)
                gen_fake_loss = self.adversarial_loss(gen_fake_feat, True, False) 
                gen_loss = 0
                imgae_loss = 10 * self.l1loss(output * (1 - mask), image * (1 - mask)) / torch.mean(1- mask) + 1 * self.l1loss(output * mask, image * mask) / torch.mean(mask) 
                
                low_loss = 10 * self.l1loss(image_low * (1 - mask), low * (1 - mask)) / torch.mean(1-mask) + 1 * self.l1loss(image_low * mask, low * mask) / torch.mean(mask) 
                high_loss = 10 * self.l1loss(image_high * (1 - mask), high * (1 - mask)) / torch.mean(1-mask) + 1 * self.l1loss(image_high * mask, high * mask) / torch.mean(mask) 
                

                gen_loss += gen_fake_loss * 0.1
                gen_loss += imgae_loss
                gen_loss += (low_loss + high_loss)

                print("[{}/{}] [{}/{}]: {} | {} | {}".format(epoch, self.args.max_epoch + 1, i, len(self.train_loader) ,gen_loss.data ,low_loss.data ,high_loss.data))
                self.optimizer_g.zero_grad()
                gen_loss.backward()
                self.optimizer_g.step()

                if i%1000==0:
                    if os.path.exists(os.path.join(self.save_path,'{}'.format(epoch//1000))):
                        pass
                    else:
                        os.makedirs(os.path.join(self.save_path,'{}'.format(epoch//1000)))
                    
                    vis = torch.cat((com_output,image)).detach().cpu()
                    vis = make_grid(vis, nrow = 8, padding = 5, normalize = True)
                    vis = T.ToPILImage()(vis)
                    vis.save(os.path.join(self.save_path,'{}/{}.jpg'.format(epoch//1000,i)))
                    
                    torch.save(self.netG.state_dict(), os.path.join(self.save_path,'Gs_0.pth'))
                    torch.save(self.netD.state_dict(), os.path.join(self.save_path,'Ds_0.pth'))
                i+=1
                
                
            self.lr_scheduler_g.step()
            self.lr_scheduler_d.step()
            
            torch.save(self.netG.state_dict(), os.path.join(self.save_path,'G_{}.pth'.format(int(epoch%15))))
            torch.save(self.netD.state_dict(), os.path.join(self.save_path,'D_{}.pth'.format(int(epoch%15))))
            
                         



class AdversarialLoss(nn.Module):
  r"""
  Adversarial loss
  https://arxiv.org/abs/1711.10337
  """

  def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
    r"""
    type = nsgan | lsgan | hinge
    """
    super(AdversarialLoss, self).__init__()
    self.type = type
    self.register_buffer('real_label', torch.tensor(target_real_label))
    self.register_buffer('fake_label', torch.tensor(target_fake_label))

    if type == 'nsgan':
      self.criterion = nn.BCELoss()
    elif type == 'lsgan':
      self.criterion = nn.MSELoss()
    elif type == 'hinge':
      self.criterion = nn.ReLU()

  def patchgan(self, outputs, is_real=None, is_disc=None):
    if self.type == 'hinge':
      if is_disc:
        if is_real:
          outputs = -outputs
        return self.criterion(1 + outputs).mean()
      else:
        return (-outputs).mean()
    else:
      labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
      loss = self.criterion(outputs, labels)
      return loss

  def __call__(self, outputs, is_real=None, is_disc=None):
    return self.patchgan(outputs, is_real, is_disc)
