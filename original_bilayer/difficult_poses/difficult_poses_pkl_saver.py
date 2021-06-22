import torch
from torch.utils import data
from torchvision import transforms
import glob
import pathlib
from PIL import Image
import numpy as np
import pickle
import cv2
import random
import math
import pdb
import os
import argparse

parser= argparse.ArgumentParser("Difficult pose finder")
parser.add_argument('--data_root',
        default= '/video-conf/scratch/pantea/random_sampled_per_person',
        type=str,
        help='dataset root')
parser.add_argument('--phase',
        type=str,
        default='train',
        help='phase of the dataset, train or test')
parser.add_argument('--results_folder',
        type=str,
        default='./results/difficult_poses',
        help='phase of the dataset, train or test')

args = parser.parse_args()

def is_difficult (keypoints):
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
                    print("danger")
                    difficulty+=1
        else:
            if left_width < right_width  and keypoints[i,0] >= keypoints[30,0]:
            #if keypoints[i,1] >= min(keypoints[30,1],keypoints[33,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[33,1]):
                print("danger!")
                difficulty+=1

        if keypoints[30,0]-keypoints[27,0] != 0:
            if left_width < right_width and keypoints[i,1]>(keypoints[30,1]-keypoints[27,1])/(keypoints[30,0]-keypoints[27,0])*(keypoints[i,0]-keypoints[27,0])+keypoints[27,1]:
                if keypoints[i,0] >= min(keypoints[30,0],keypoints[33,0]) and keypoints[i,0] <= max(keypoints[30,0],keypoints[33,0]):
                #if keypoints[i,1] >= min(keypoints[30,1],keypoints[27,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[27,1]):
                    print("danger!")
                    difficulty+=1
        else:
            if left_width < right_width  and keypoints[i,0] >= keypoints[27,0]:
            #if keypoints[i,1] >= min(keypoints[30,1],keypoints[27,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[27,1]):
                print("danger!")
                difficulty+=1

    if difficulty >0:
        return True
    
    return False

def load_pickle(path_string):
    pkl_file = open(path_string, 'rb')
    my_dict = pickle.load(pkl_file)
    pkl_file.close()
    return my_dict

data_root = args.data_root
phase = args.phase
dataset_name =  str(data_root).split('/')[-1:][0]
main_dir = str(args.results_folder) + '/' + str(dataset_name)

keypoint_directory=pathlib.Path(data_root+'/keypoints/'+phase)
print(keypoint_directory)
# Find all the video sessions
keypoints_sessions = keypoint_directory.glob('*/*/*')
keypoints_sessions = sorted([str(seq) for seq in keypoints_sessions])


for session in keypoints_sessions:
    session_relative_name= '/'.join(str(session).split('/')[-4:])
    session_directory = pathlib.Path(str(session))
    keypoints_paths = session_directory.glob('*')
    keypoints_paths = sorted([str(seq) for seq in keypoints_paths])
    #print(session, keypoints_paths)
    save_dir = main_dir + '/' + session_relative_name
    os.makedirs(str(save_dir), exist_ok=True)
    mydict = {}
    counter = 0
    mydict [('-1')]= str(data_root)
    for i in range(0, len(keypoints_paths)-1):
        print(i)
        keypoints1 = np.load(keypoints_paths[i])
        flag = is_difficult(keypoints1)
        if flag:
            name_1 = '/'.join(str(keypoints_paths[i]).split('/')[-1:])
            name_1= name_1.split('.')[0]
            mydict[(str(counter))] = name_1
            counter+=1
    pickle.dump(mydict,  open(str(save_dir) + "/" + 'Difficult_poses' + '.pkl', 'wb'))
    print(session, "completed!")

