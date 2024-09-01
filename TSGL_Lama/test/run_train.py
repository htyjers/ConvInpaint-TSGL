""" Generate commands for meta-train phase. """
import os
import math
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 114
def run_exp():    
    the_command = (
        'python3 TSGL_Lama/test/main.py'
        + ' --dataset_dir=' + 'dataset_dir'
        + ' --maskdataset_dir=' + 'maskdataset_dir'
    )

    os.system(the_command + ' --phase=train')

run_exp()          ## best 
