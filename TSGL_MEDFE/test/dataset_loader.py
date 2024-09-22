""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from canny import image_to_edge
import random

class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args):
        # Set the path according to train, val and test   
        self.args = args
        THE_PATH =  args.dataset_dir    ##args.dataset_dir
        data = []       
        data1 = []
        for root, dirs, files in os.walk(THE_PATH, topdown=True):
            for name in files:
                data1.append(osp.join(root, name))
                root1 = root.replace('RTV','RGB')
                data.append(osp.join(root1, name))
                
        self.data = data   
        print(len(self.data))
        self.data1 = data1    
        self.image_size = args.image_size

        
    def __len__(self):
        return len(self.data)


    
    def __getitem__(self, i):
        path = self.data[i]
        path1 = self.data1[i]
#         print(path)
        image = Image.open(path).convert('RGB')
        low = Image.open(path1).convert('RGB')
        
        image, low = define_transformer(image,low)
        high,gray = image_to_edge(image, sigma=2.)
        
        return image, high, low, gray


def define_transformer(image, low):
    transform_list = []
    
    w,h = image.size
    x = random.randint(0, np.maximum(0, w - 256))
    y = random.randint(0, np.maximum(0, h - 256))
    crop = [x, y, 256, 256]
    crop_position = crop[:2]  
    crop_size = crop[2:]  
    # For PSV and Places2
    transform_list.append(transforms.Lambda(lambda img: __crop(img, crop_position, crop_size)))
    # For Celeba 
    transform_list.append(transforms.Resize(size=(256, 256), interpolation=Image.NEAREST))    
    
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    
    trans = transforms.Compose(transform_list)
    image = trans(image)
    low = trans(low)
    return image,low

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw, th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return 
