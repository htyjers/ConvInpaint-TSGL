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
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from dataset_loader import DatasetLoader as Dataset
from dataset_loader1 import DatasetLoader as Dataset1
from itertools import cycle
import random
import warnings
from trans import feature2token
from model import Generator
#import sklearn.metrics as skm

import torch.nn.functional as F
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

        self.trainset = Dataset('test', self.args)
        self.trainset1 = Dataset1('train', self.args)
        
        self.model = Generator().to(self.args.device)
        self.model = torch.nn.DataParallel(self.model)
        
    def train(self):

        self.model.load_state_dict(torch.load("pretrain_path"))
        self.model.eval() 

        self.train_loader = DataLoader(self.trainset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)
        self.train_loader1 = DataLoader(self.trainset1, batch_size=1, shuffle=False, num_workers=1, drop_last=True) 

        for epoch in range(0,1):
            mask_iterator = iter(self.train_loader1)
            i = 0
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
                
                with torch.no_grad():
                        output = self.model(image * mask, high * mask, low * mask, gray * mask, mask)
                        vis = (image * mask + output * (1 - mask)).detach().cpu()#
                        vis = make_grid(vis, nrow = 1, padding = 0, normalize = True)
                        vis = T.ToPILImage()(vis)
                        vis.save(os.path.join(self.save_path,'{}_f.jpg'.format(i)))

                        vis = (image).detach().cpu()#
                        vis = make_grid(vis, nrow = 1, padding = 0, normalize = True)
                        vis = T.ToPILImage()(vis)
                        vis.save(os.path.join(self.save_path,'{}_r.jpg'.format(i)))
                
                i+=1
#                 
