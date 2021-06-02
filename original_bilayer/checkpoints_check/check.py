# Since we tried to freeze some of the networks, to check wether the corresponding networks are actually frozen, this file manually loads the 
# checkpoints and compares them. 


import sys
sys.path.append('../')
import torch
import glob
import copy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import time
import argparse

checkpoints_dir = '/video-conf/scratch/pantea_experiments_mapmaker/runs/all_networks_frozen_except_inference_generator/checkpoints'
checkpoint_path1 = checkpoints_dir+'/'+'100_texture_generator.pth'
checkpoint_path2=  checkpoints_dir+'/'+'1100_texture_generator.pth'
def match_checkpoints(checkpoint_path1, checkpoint_path2):
    unmatched_keys = []
    checkpoint1 = torch.load(checkpoint_path1, map_location='cpu')
    checkpoint2 = torch.load(checkpoint_path2, map_location='cpu')

    for key in checkpoint1.keys():

        if torch.all(torch.eq(checkpoint1[key], checkpoint2[key])):
            print("key", key, "matches.")
        else:
            unmatched_keys.append(key)
            print(key)
            print("diff:", torch.abs(checkpoint1[key]-checkpoint2[key]).sum())
    return unmatched_keys

def print_values (checkpoint_path1):
    checkpoint1 = torch.load(checkpoint_path1, map_location='cpu')

    for key in checkpoint1.keys():
        key_split = key.split(".")
        if key_split[0]=='gen_tex' and  key_split[1]=='heads': 
            print(key, checkpoint1[key], checkpoint1[key].shape)

unmatched_keys = match_checkpoints(checkpoint_path1, checkpoint_path2)
print("Which keys don't match?", str(unmatched_keys))
