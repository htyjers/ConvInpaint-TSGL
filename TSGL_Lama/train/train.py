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

from dataset_loader_lama import DatasetLoader as Dataset
from dataset_loader1 import DatasetLoader as Dataset1
from model import FFCResNetGenerator,Discriminator
from itertools import cycle
from pcp import ResNetPL
from losses import *
import random
import warnings
import utils
from utils import *
from utils import stitch_images
warnings.filterwarnings("ignore")
class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
class Trainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = '/data/512/inpaint_log'
        meta_base_dir = osp.join(log_base_dir, args.file_name)
        self.save_path = meta_base_dir
        if os.path.exists(self.save_path):
            pass
        else:
            os.makedirs(self.save_path)
        self.args = args

        self.trainset = Dataset('train', self.args)
        self.trainset1 = Dataset1('train', self.args)
        self.netG = FFCResNetGenerator().to(self.args.device)
        self.netD = Discriminator().to(self.args.device)
        
        
        self.l1loss = nn.L1Loss()
        self.loss_resnet_pl = ResNetPL().to(self.args.device)
        
        self.optimizer_g = torch.optim.Adam(self.netG.parameters(), lr = 1e-3, betas = (0, 0.99))
        self.optimizer_d = torch.optim.Adam(self.netD.parameters(), lr = 1e-4, betas = (0, 0.99))
         
        self.netG = torch.nn.DataParallel(self.netG)
        self.netD = torch.nn.DataParallel(self.netD)
        

    def train(self):
        self.netG.train()
        self.netD.train()
        self.train_loader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        self.train_loader1 = DataLoader(self.trainset1, batch_size=self.args.batch_size, shuffle=True, num_workers=8, drop_last=True) 
        for epoch in range(0, self.args.max_epoch + 1):
            mask_iterator = iter(self.train_loader1)
            i = 0
            for (image, high, gray) in self.train_loader:
                image = image.to(self.args.device)
                high = high.to(self.args.device)
                gray = gray.to(self.args.device)
                #'''
                try:
                    mask = next(mask_iterator).to(self.args.device)
                except StopIteration:
                    mask_iterator = iter(self.train_loader1)
                    mask = next(mask_iterator).to(self.args.device)
                #'''
                B,C,H,W = image.size()
                
                self.optimizer_g.zero_grad()

                output,image_high= self.netG(image * mask, high * mask, gray * mask, 1-mask) #
                com_output = image * mask + output * (1 - mask)
                gen_dis, gen_feats = self.netD(output)
                real_dis, real_feats = self.netD(image)

                
                # Generator Loss
                high_loss = (10 * self.l1loss(image_high * mask, high * mask) + self.l1loss(image_high * (1 - mask), high * (1 - mask))).mean()
                imgae_loss = (10 * self.l1loss(output * mask, image * mask)).mean()
                pcp_loss = self.loss_resnet_pl(output, image) * 10
                fm_loss = feature_matching_loss(gen_feats, real_feats, mask=None) * 200
                adv_gen_loss = generator_loss(discr_fake_pred=gen_dis, mask=1-mask)
                gen_loss = imgae_loss + pcp_loss + fm_loss + adv_gen_loss + high_loss# + low_loss
                gen_loss.mean().backward()

                self.optimizer_g.step()

                # Discriminator loss
                self.optimizer_d.zero_grad()

                real_img_tmp = image.detach().requires_grad_(True)
                real_logits, _ = self.netD(real_img_tmp)
                gen_logits, _ = self.netD(output.detach())
                dis_real_loss, grad_penalty = discriminator_real_loss(real_batch=real_img_tmp, discr_real_pred=real_logits,
                                                                      gp_coef=0.001, do_GP=True)
                dis_fake_loss = discriminator_fake_loss(discr_fake_pred=gen_logits, mask=1-mask)
                dis_loss = dis_real_loss + dis_fake_loss + grad_penalty

                dis_loss.mean().backward()

                self.optimizer_d.step()
                print("[{}/{}] [{}/{}]: {} | {} | {} | {} | {} -- {}".format(epoch, self.args.max_epoch + 1, i, len(self.train_loader) ,imgae_loss.data ,pcp_loss.data ,fm_loss.data ,adv_gen_loss.data ,high_loss.data,gen_loss.data))#
                    
                
                with torch.no_grad():
                    if i%200==0:
                        if os.path.exists(os.path.join(self.save_path,'{}'.format(epoch//100))):
                            pass
                        else:
                            os.makedirs(os.path.join(self.save_path,'{}'.format(epoch//100)))
                    
                        vis = torch.cat((output,com_output,image)).detach().cpu()
                        vis = make_grid(vis, nrow = B//2, padding = 5, normalize = False)
                        vis = T.ToPILImage()(vis)
                        vis.save(os.path.join(self.save_path,'{}/a_{}.jpg'.format(epoch//100,i)))
                    
                        torch.save(self.netG.state_dict(), os.path.join(self.save_path,'Gs_0.pth'))
                        torch.save(self.netD.state_dict(), os.path.join(self.save_path,'Ds_0.pth'))
                i+=1

            torch.save(self.netG.state_dict(), os.path.join(self.save_path,'G_{}.pth'.format(int(epoch))))
            torch.save(self.netD.state_dict(), os.path.join(self.save_path,'D_{}.pth'.format(int(epoch))))
            


