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

checkpoints_dir = '/video-conf/scratch/pantea_experiments_mapmaker/runs/No_frozen_network_new_keypoints_from_pretrained/checkpoints'
checkpoint_path1 = checkpoints_dir+'/'+'100_texture_generator.pth'


#checkpoint_path2=  checkpoints_dir+'/'+'1100_texture_generator.pth'
def match_checkpoints(checkpoint_path1, checkpoint_path2):
    unmatched_keys = []
    checkpoint1 = torch.load(checkpoint_path1, map_location='cpu')
    metadata = getattr(checkpoint1, '_metadata', None)
    #print("checkpoint1 :",metadata, '\n\n')
    checkpoint2 = torch.load(checkpoint_path2, map_location='cpu')
    metadata = getattr(checkpoint2, '_metadata', None)
    #print("checkpoint2 :",metadata, '\n\n')
    #print(checkpoint.keys(), checkpoint.values())
    for key in checkpoint1.keys():

        if torch.all(torch.eq(checkpoint1[key], checkpoint2[key])):
            pass
            #print("key", key, "matches.")
        else:
            unmatched_keys.append(key)
            #if key == 'gen_tex.blocks.3.block.0.weight_v':
            #print("Found unmatched keys!", key, '\n\n')
            print(key)
            print("diff:", torch.abs(checkpoint1[key]-checkpoint2[key]).sum())
            #print("second:",checkpoint2[key])
            #return False
    return unmatched_keys

def print_values (checkpoint_path1):
    checkpoint1 = torch.load(checkpoint_path1, map_location='cpu')

    for key in checkpoint1.keys():
        key_split = key.split(".")
        if key_split[0]=='gen_tex' and  key_split[1]=='heads': 
            print(key, checkpoint1[key], checkpoint1[key].shape)

print_values(checkpoint_path1)
