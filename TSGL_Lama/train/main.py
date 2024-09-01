""" Main function for this repo. """
import argparse
import numpy as np
import torch
from train import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument('--num_work', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--dataset', type=str, default='paris') # Dataset
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test']) # Phase
    parser.add_argument('--seed', type=int, default=0) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--dataset_dir', type=str, default=None) # Dataset folder
    parser.add_argument('--maskdataset_dir', type=str, default=None) # Dataset folder

    # model parameters
    parser.add_argument('--gan_type', type=str, default='hinge')


    # training pamameters
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--start_epoch', type=int, default=0)

    # log parameter
    parser.add_argument('--file_name', type=str, default='train')#
    parser.add_argument('--meta_label', type=str, default='exp1')
    parser.add_argument('--index', type=int, default=0)

    # Set and print the parameters
    args = parser.parse_args()
    print(args)

    # Set manual seed for PyTorch
    if args.seed==0:
        print ('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print ('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Start trainer for train or test
    if args.phase=='train':
        trainer = Trainer(args)
        trainer.train()
