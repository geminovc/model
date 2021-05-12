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
checkpoint_path1 = checkpoints_dir+'/'+'100_identity_embedder.pth'
checkpoint_path2=  checkpoints_dir+'/'+'1000_identity_embedder.pth'
def match_checkpoints(checkpoint_path1, checkpoint_path2):
    unmatched_keys = []
    checkpoint1 = torch.load(checkpoint_path1, map_location='cpu')
    checkpoint2 = torch.load(checkpoint_path2, map_location='cpu')
    #print(checkpoint.keys(), checkpoint.values())
    for key in checkpoint1.keys():

        if torch.all(torch.eq(checkpoint1[key], checkpoint2[key])):
            print("key", key, "matches.")
        else:
            unmatched_keys.append(key)
            print("Found unmatched keys!", key)
            print("first:",checkpoint1[key])
            print("second:",checkpoint2[key])
            #return False
    return unmatched_keys



unmatched_keys = match_checkpoints(checkpoint_path1, checkpoint_path2)
print("Which keys don't match?", str(unmatched_keys))
print("All the keys?", torch.load(checkpoint_path1, map_location='cpu').keys())