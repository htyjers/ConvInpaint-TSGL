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
import torchvision.utils as tvutils

from dataset_loader_lama import DatasetLoader as Dataset
from dataset_loader1 import DatasetLoader as Dataset1
from model import FFCResNetGenerator
import cv2
import warnings
warnings.filterwarnings("ignore")
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

        self.netG = torch.nn.DataParallel(self.netG)
    def train(self):
        self.netG.load_state_dict(torch.load("pretrain_path"))
        self.netG.eval()
        self.train_loader = DataLoader(self.trainset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
        self.train_loader1 = DataLoader(self.trainset1, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
        for epoch in range(0, 1):
            mask_iterator = iter(self.train_loader1)
            i = 0
            for (image, high, gray) in self.train_loader:
                image = image.to(self.args.device)
                high = high.to(self.args.device)
                gray = gray.to(self.args.device)
                try:
                    mask = next(mask_iterator).to(self.args.device)
                except StopIteration:
                    mask_iterator = iter(self.train_loader1)
                    mask = next(mask_iterator).to(self.args.device)

                with torch.no_grad():
                    output,high1= self.netG(image * mask, high * mask, gray * mask, 1-mask) #,high1
                    img = (image * mask + output * (1 - mask)) * 255
                    img = img.permute(0, 2, 3, 1).int().cpu().numpy()
                    cv2.imwrite(self.save_path+'/{}_f.jpg'.format(i), img[0, :, :, ::-1])

                    img = (image) * 255
                    img = img.permute(0, 2, 3, 1).int().cpu().numpy()
                    cv2.imwrite(self.save_path+'/{}_r.jpg'.format(i), img[0, :, :, ::-1])

                    i+=1

