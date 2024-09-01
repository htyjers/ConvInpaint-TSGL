""" Generate commands for meta-train phase. """
import os
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7,8' # 114
def run_exp():    
    the_command = (
        'python3 TSGL_MEDFE/test/main.py'
        + ' --dataset_dir=' + 'dataset_dir'
        + ' --maskdataset_dir=' + 'maskdataset_dir'
    )

    os.system(the_command + ' --phase=train')

run_exp()          ## best 
