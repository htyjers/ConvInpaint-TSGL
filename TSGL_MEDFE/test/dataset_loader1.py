""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args):
        # Set the path according to train, val and test   
        self.args = args
        THE_PATH = args.maskdataset_dir
        print(THE_PATH)
        # Generate empty list for data and label           
        data = []
        for root, dirs, files in os.walk(THE_PATH, topdown=True):
            for name in files:
                data.append(osp.join(root, name))
                
        self.data = data    
        print(len(self.data))
        self.image_size = args.image_size

        self.transform = transforms.Compose([
        	transforms.Resize(size=(256, 256), interpolation=Image.NEAREST),
        	transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = self.data[i]
#         print(path)
        mask = self.transform(Image.open(path).convert('1'))
        return 1 - mask
