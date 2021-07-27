"""
This script validates that the networks intended to be frozen are frozen by 
comparing checkpoints across epochs. Prints out the matching checkpoints and differing ones.

Sample usage:

python compare_checkpoints.py --checkpoints_dir <directory_to_the_checkpoints>  
--net_name <name_of_the_network> --first_epoch 100 --second_epoch 1000
"""

#Importing libraries
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

parser= argparse.ArgumentParser("Checkpoint compare")
parser.add_argument('--checkpoints_dir',
        type=str,
        default='/video-conf/scratch/pantea_experiments_mapmaker/runs/all_networks_frozen_except_inference_generator/checkpoints',
        help='the directory of the two checkpoints that you want to compare. ')
parser.add_argument('--net_name',
        type=str,
        default='texture_generator',
        help='the network that you want to check')
parser.add_argument('--first_epoch',
        type=int,
        default=100,
        help='the first epoch for comparison')
parser.add_argument('--second_epoch',
        type=int,
        default=1100,
        help='the second epoch for comparison')

args = parser.parse_args()


checkpoints_dir = args.checkpoints_dir
net_name = args.net_name
checkpoint_path1 = checkpoints_dir + '/' + str(args.first_epoch) +'_' + net_name + '.pth'
checkpoint_path2=  checkpoints_dir + '/' + str(args.second_epoch)+'_' + net_name + '.pth'

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
