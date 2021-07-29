"""
This file generates .pkl files from datasets with the voxceleb2 dataset structure, and stores them in the voxceleb2 dataset structure.
Each .pkl file contains the frame number of images with left_tilted difficult poses in each session.    
A difficult left_tilted pose is a pose where the nose covers the cheeks behind it.  

The main voxceleb dataset structure is in the format of:
DATA_ROOT/[imgs, keypoints, segs]/[train, test]/PERSON_ID/VIDEO_ID/SEQUENCE_ID/FRAME_NUM[.jpg, .npy, .png]

This file reads all the DATA_ROOT/keypoints/[train, test]/PERSON_ID/VIDEO_ID/SEQUENCE_ID/*
and stores the frame number of images with difficult poses in the format of:
RESULTS_FOLDER/DATASET_NAME/[train, test]/PERSON_ID/VIDEO_ID/SEQUENCE_ID/Difficult_poses.pkl


To run this file use:

python find_difficult_poses.py --data_root <'PATH_TO_YOUR_DATA_ROOT'>
--phase <'train' or 'test'>  --results_folder <'PATH_TO_WHERE_YOU_WANT_TO_SAVE_PKL_FILES'> 

Example:
python find_difficult_poses.py --data_root '/video-conf/scratch/pantea/temp_per_person_extracts' 
--phase 'test' --results_folder './result'

Inputs
----------

Inputs are set in parser file as the following:

    --data_root <'PATH_TO_YOUR_DATA_ROOT'>
    --phase <'train' or 'test'>  
    --results_folder <'PATH_TO_WHERE_YOU_WANT_TO_SAVE_PKL_FILES'> 
    --min_per_session_keypoints <minimum number of frames per session for collecting images later in the training pipeline>

The data_root should contain files in the format of:

DATA_ROOT/[imgs, keypoints, segs]/[train, test]/PERSON_ID/VIDEO_ID/SEQUENCE_ID/FRAME_NUM[.jpg, .npy, .png]

For the example of the data_root structure refer to ../dataset/voxceleb2.py documentation. 

The results_folder is the root to save difficult_pose .pkl files. The code will automatically extract the dataset name such as per_person, per_video, general, etc. 
and stores the corresponding pairwise distances in path RESULTS_FOLDER/DATASET_NAME. 

Outputs
----------

The output is a dataset in the format of: 
RESULTS_FOLDER/DATASET_NAME/[train, test]/PERSON_ID/VIDEO_ID/SEQUENCE_ID/Difficult_poses.pkl

Each Difficult_poses.pkl file has the following structure:

{
('-1'): 'data_root of the dataset that the keypoints belong to',
('0'): 'frame_num',
..

}

"""
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

parser = argparse.ArgumentParser("Difficult pose finder")

parser.add_argument('--data_root',
        default= '/video-conf/scratch/pantea/per_video_1_three_datasets',
        type=str,
        help='dataset root')

parser.add_argument('--phase',
        type=str,
        default='test',
        help='phase of the dataset, train or test')

parser.add_argument('--results_folder',
        type=str,
        default='/data/pantea/pose_results/difficult_poses',
        help='phase of the dataset, train or test')

parser.add_argument('--min_per_session_keypoints',
        type=int,
        default=2,
        help='each pickle file should contain at least a number of keypoints')

args = parser.parse_args()

# The function determines if a pose -shown by the keypoints- is difficult 
# A difficult pose is a pose where the nose covers the cheeks behind it.  
def is_difficult (keypoints):
    keypoints = keypoints.reshape((68,2))
    # keypoints [30]: nose tip
    # keypoints [33]: connection of nose and philtrum
    # keypoints [0] to keypoints [4]: right cheek
    # keypoints [12] to keypoints [16]: left cheek
    # keypoints[27]: left most side of the left eyebrow 

    # left_width is an estimator of the width of the left side of the face
    # right_width is an estimator of the width of the right side of the face
    left_width = (keypoints[30,1] - keypoints[2,1])**2 + (keypoints[30,0] - keypoints[2,0])**2
    right_width = (keypoints[30,1] - keypoints[14,1])**2 + (keypoints[30,0] - keypoints[14,0])**2
    difficulty = 0

    for i in range(1,4):
        if keypoints[30,0] - keypoints[33,0] != 0:
            m = (keypoints[30,1] - keypoints[33,1]) / (keypoints[30,0] - keypoints[33,0])
            if left_width < right_width and keypoints[i,1] < m * (keypoints[i,0] - keypoints[33,0]) + keypoints[33,1]:
                if keypoints[i,0] >= min(keypoints[30,0], keypoints[33,0]) and keypoints[i,0] <= max(keypoints[30,0], keypoints[33,0]):
                    print("found difficult pose!")
                    difficulty += 1
        else:
            if left_width < right_width  and keypoints[i,0] >= keypoints[30,0]:
                print("found difficult pose!!")
                difficulty += 1

        if keypoints[30,0] - keypoints[27,0] != 0:
            m = (keypoints[30,1] - keypoints[27,1]) / (keypoints[30,0] - keypoints[27,0])
            if left_width < right_width and keypoints[i,1]> m * (keypoints[i,0] - keypoints[27,0]) + keypoints[27,1]:
                if keypoints[i,0] >= min(keypoints[30,0], keypoints[33,0]) and keypoints[i,0] <= max(keypoints[30,0], keypoints[33,0]):
                    print("found difficult pose!!")
                    difficulty += 1
        else:
            if left_width < right_width  and keypoints[i,0] >= keypoints[27,0]:
                print("found difficult pose!!")
                difficulty += 1

    if difficulty > 0:
        return True
    
    return False


data_root = args.data_root
phase = args.phase
dataset_name =  str(data_root).split('/')[-1:][0]
main_dir = str(args.results_folder) + '/' + str(dataset_name)

# Going through all the keypoints to find the difficult ones
keypoint_directory = pathlib.Path(data_root + '/keypoints/' + phase)
print(keypoint_directory)

# Find all the video sessions
keypoints_sessions = keypoint_directory.glob('*/*/*')
keypoints_sessions = sorted([str(seq) for seq in keypoints_sessions])
number_of_difficult_poses = 0

for session in keypoints_sessions:
    session_relative_name = '/'.join(str(session).split('/')[-4:])
    session_directory = pathlib.Path(str(session))
    keypoints_paths = session_directory.glob('*')
    keypoints_paths = sorted([str(seq) for seq in keypoints_paths])
    save_dir = main_dir + '/' + session_relative_name
    mydict = {}
    counter = 0
    mydict [('-1')] = str(data_root)
    for i in range(0, len(keypoints_paths) - 1):
        print(i)
        keypoints1 = np.load(keypoints_paths[i])
        flag = is_difficult(keypoints1)
        if flag:
            name_1 = '/'.join(str(keypoints_paths[i]).split('/')[-1:])
            name_1 = name_1.split('.')[0]
            mydict[(str(counter))] = name_1
            counter += 1
            number_of_difficult_poses += 1
    
    if len(mydict) >= 1 + args.min_per_session_keypoints:
        print("Saving the pickle for session!")
        os.makedirs(str(save_dir), exist_ok=True)
        pickle.dump(mydict,  open(str(save_dir) + "/" + 'Difficult_poses' + '.pkl', 'wb'))
    else:
        print("This session did not have enough difficult poses!")
    
    print(session, "completed!")

print("percentage of difficult poses: ", str(number_of_difficult_poses/len(keypoints_paths)))