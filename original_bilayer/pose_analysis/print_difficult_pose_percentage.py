"""
This file finds the percentage of left-tilted difficult poses in a keypoint directory.
Difficult poses are poses where the cheeks are behind the nose and you can not see a part of them.

To run this file use:

python print_difficult_pose_percentage.py --keypoint_directory <'KEYPOINTS_DIR'>

Example:
python print_difficult_pose_percentage.py --keypoint_directory '/video-conf/scratch/pantea/temp_per_person_extracts/keypoints/train'

Inputs
----------
keypoint_directory: path to keypoints

Outputs
----------

Prints the percentage of difficult left-tilted poses.


"""


import torch
from torch.utils import data
from torchvision import transforms
import glob
import pathlib
from PIL import Image
import numpy as np
import pickle as pkl
import cv2
import random
import math
import pdb
import os

parser= argparse.ArgumentParser("Difficult pose finder")
parser.add_argument('--keypoint_directory',
        default= '/video-conf/scratch/pantea/temp_per_person_extracts/keypoints/train',
        type=str,
        help='keypoints directory')


args = parser.parse_args()

def normalizeAngle(angle):
    newAngle = angle
    if (newAngle <= -np.pi):
         newAngle += 2*np.pi
    if (newAngle > np.pi):
         newAngle -= 2*np.pi
    return newAngle


keypoint_directory=pathlib.Path(args.keypoint_directory)
keypoints_paths = keypoint_directory.glob('*/*/*/*')
keypoints_paths = sorted([str(seq) for seq in keypoints_paths])

number_of_difficult_poses = 0
for keypoint_path in keypoints_paths:
    print(keypoint_path) 
    keypoints = np.load(keypoint_path).astype('float32')
    keypoints = keypoints.reshape((68,2)) 


    left_width = (keypoints[30,1]-keypoints[2,1])**2 + (keypoints[30,0]-keypoints[2,0])**2
    right_width = (keypoints[30,1]-keypoints[14,1])**2 + (keypoints[30,0]-keypoints[14,0])**2
    difficulty = 0
    
    # left-tilted pose
    for i in range(1,4):
        if keypoints[30,0]-keypoints[33,0] != 0:
            if left_width < right_width and keypoints[i,1]<(keypoints[30,1]-keypoints[33,1])/(keypoints[30,0]-keypoints[33,0])*(keypoints[i,0]-keypoints[33,0])+keypoints[33,1]:
                if keypoints[i,0] >= min(keypoints[30,0],keypoints[33,0]) and keypoints[i,0] <= max(keypoints[30,0],keypoints[33,0]):
                #if keypoints[i,1] >= min(keypoints[30,1],keypoints[33,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[33,1]):
                    print("found difficult pose")
                    difficulty+=1
        else:
            if left_width < right_width  and keypoints[i,0] >= keypoints[30,0]:
            #if keypoints[i,1] >= min(keypoints[30,1],keypoints[33,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[33,1]):
                print("found difficult pose!")
                difficulty+=1

        if keypoints[30,0]-keypoints[27,0] != 0:
            if left_width < right_width and keypoints[i,1]>(keypoints[30,1]-keypoints[27,1])/(keypoints[30,0]-keypoints[27,0])*(keypoints[i,0]-keypoints[27,0])+keypoints[27,1]:
                if keypoints[i,0] >= min(keypoints[30,0],keypoints[33,0]) and keypoints[i,0] <= max(keypoints[30,0],keypoints[33,0]):
                #if keypoints[i,1] >= min(keypoints[30,1],keypoints[27,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[27,1]):
                    print("found difficult pose!")
                    difficulty+=1
        else:
            if left_width < right_width  and keypoints[i,0] >= keypoints[27,0]:
            #if keypoints[i,1] >= min(keypoints[30,1],keypoints[27,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[27,1]):
                print("found difficult pose!")
                difficulty+=1

    if difficulty >0:
        number_of_difficult_poses+=1

print("percentage of difficult poses: ", str(number_of_difficult_poses/len(keypoints_paths)))