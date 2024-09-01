""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from canny import image_to_edge
import random
import glob
import logging
import os
import random
from torchvision import transforms as t

class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args):
        # Set the path according to train, val and test   
        self.data = list(glob.glob(os.path.join(args.dataset_dir, '**', '*.jpg'), recursive=True))
        
        print(len(self.data))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = self.data[i]
        image = Image.open(path).convert('RGB')
        image = define_transformer(image)
        high,gray = image_to_edge(image, sigma=2.)
        return image, high, gray

def define_transformer(image):
    transform_list = []
    #train
    #'''
    w,h = image.size
    x = random.randint(0, np.maximum(0, w - 256))
    y = random.randint(0, np.maximum(0, h - 256))
    crop = [x, y, 256, 256]
    crop_position = crop[:2]  
    crop_size = crop[2:]  
    transform_list.append(transforms.Lambda(lambda img: __crop(img, crop_position, crop_size)))

    if random.random() > 0.5:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, True)))

    transform_list += [transforms.ToTensor()]
    
    trans = transforms.Compose(transform_list)
    image = trans(image)
    return image

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw, th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return 
