"""
This file generates .pkl files from datasets with the voxceleb2 dataset structure, and stores them in the voxceleb2 dataset structure.
Each .pkl file contains the pairwise L2 distance between corresponding keypoints in each session.    

The main voxceleb dataset structure is in the format of:
DATA_ROOT/[imgs, keypoints, segs]/[train, test]/PERSON_ID/VIDEO_ID/SEQUENCE_ID/FRAME_NUM[.jpg, .npy, .png]

This file reads all the 

DATA_ROOT/keypoints/[train, test]/PERSON_ID/VIDEO_ID/SEQUENCE_ID/*

and stores the pairwise L2 distance of each two keypoints and stores them in the format of:
RESULTS_FOLDER/DATASET_NAME/[train, test]/PERSON_ID/VIDEO_ID/SEQUENCE_ID/L2_Distaces.pkl


To run this file use:

python find_l2_distance.py --data_root <'PATH_TO_YOUR_DATA_ROOT'>
--phase <'train' or 'test'>  --results_folder <'PATH_TO_WHERE_YOU_WANT_TO_SAVE_PKL_FILES'> 

Example:
python find_l2_distance.py --data_root '/video-conf/scratch/pantea/temp_per_person_extracts' 
--phase 'test' --results_folder './result'

Inputs
----------

Inputs are set in parser file as the following:

    --data_root <'PATH_TO_YOUR_DATA_ROOT'>
    --phase <'train' or 'test'>  
    --results_folder <'PATH_TO_WHERE_YOU_WANT_TO_SAVE_PKL_FILES'> 

The data_root should contain files in the format of:

DATA_ROOT/[imgs, keypoints, segs]/[train, test]/PERSON_ID/VIDEO_ID/SEQUENCE_ID/FRAME_NUM[.jpg, .npy, .png]

Example of the data_root structure:

                 DATA_ROOT - [imgs, keypoints, segs] _ phase _ id00012 _ abc _ 00001 _ 0 [.jpg, .npy, .png]
                                                            |         |            |_ 1 [.jpg, .npy, .png]
                                                            |         |            |_ ...
                                                            |         |            |_ 99 [.jpg, .npy, .png]
                                                            |         |
                                                            |         |_ def  _ 00001 _ 0 [.jpg, .npy, .png]
                                                            |                |       |_ 1 [.jpg, .npy, .png]
                                                            |                |       |_ ...
                                                            |                |       |_ 150 [.jpg, .npy, .png]
                                                            |                |
                                                            |                |_ 00002 _ 0 [.jpg, .npy, .png]
                                                            |                        |_ 1 [.jpg, .npy, .png]
                                                            |                        |_ ... 
                                                            |                        |_ 89 [.jpg, .npy, .png]
                                                            |               
                                                            |_ id00013 _ lmn _ 00001 _ 0 [.jpg, .npy, .png]
                                                            |          |             |_ 1 [.jpg, .npy, .png]
                                                            |          |             |_ ... 
                                                            |          |             |_ 89 [.jpg, .npy, .png]
                                                            |          |
                                                            |          |_ opq  _ 00001 _ ...
                                                            |                 |_ 00002 _ ...
                                                            |                 |_ 00003 _ ...
                                                            |
                                                            |_ id00014 _ rst _ 00001 _ ...
                                                                        |    |_ 00002 _ ...
                                                                        |
                                                                        |_ uvw  _ 00001 _ 0 [.jpg, .npy, .png]
                                                                                |       |_ 1 [.jpg, .npy, .png]
                                                                                |       |_ ... 
                                                                                |       |_ 68 [.jpg, .npy, .png]
                                                                                |
                                                                                |_ 00002 _ 0 [.jpg, .npy, .png]
                                                                                |       |_ ...
                                                                                |       |_ 299 [.jpg, .npy, .png]
                                                                                |
                                                                                |_ 00003 _ 0 [.jpg, .npy, .png]
                                                                                        |_ ...
                                                                                        |_ 100 [.jpg, .npy, .png]

The results_folder is the root to save l2_distance .pkl files. The code will automatically extract the dataset name such as per_person, per_video, general, etc. 
and stores the corresponding pairwise distances in path RESULTS_FOLDER/DATASET_NAME. 

Outputs
----------

The output is a dataset in the format of: 
RESULTS_FOLDER/DATASET_NAME/[train, test]/PERSON_ID/VIDEO_ID/SEQUENCE_ID/L2_Distances.pkl

Example of the output structure:

                RESULTS_FOLDER - DATASET_NAME _ phase _ id00012 _ abc _ 00001 _ L2_Distances.pkl
                                                    |         |            
                                                    |         |           
                                                    |         |            
                                                    |         |
                                                    |         |_ def  _ 00001 _ L2_Distances.pkl
                                                    |                |       
                                                    |                |     
                                                    |                |       
                                                    |                |
                                                    |                |_ 00002 _ L2_Distances.pkl
                                                    |                        
                                                    |                      
                                                    |                        
                                                    |               
                                                    |_ id00013 _ lmn _ 00001 _ L2_Distances.pkl
                                                    |          |            
                                                    |          |       
                                                    |          |           
                                                    |          |
                                                    |          |_ opq  _ 00001 _ ...
                                                    |                 |_ 00002 _ ...
                                                    |                 |_ 00003 _ ...
                                                    |
                                                    |_ id00014 _ rst _ 00001 _ ...
                                                                |    |_ 00002 _ ...
                                                                |
                                                                |_ uvw  _ 00001 _ L2_Distances.pkl
                                                                        |      
                                                                        |     
                                                                        |      
                                                                        |
                                                                        |_ 00002 _ L2_Distances.pkl
                                                                        |     
                                                                        |     
                                                                        |
                                                                        |_ 00003 _L2_Distances.pkl
                                                                              
                                                                               

Each L2_Distances.pkl file contains a dictionary with the following structure:

{
('-1','-1'): 'data_root of the dataset that the keypoints belong to',
('frame_num_1','frame_num_1'): 'L2_distance (keypoint_1, keypoint_2)',
...
}

"""
# Importing libraries
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

# Adding the parser
parser= argparse.ArgumentParser("L2 distance finder")
parser.add_argument('--data_root',
        default= '/video-conf/scratch/pantea/temp_per_person_extracts',
        type=str,
        help='dataset root')
parser.add_argument('--phase',
        type=str,
        default='test',
        help='phase of the dataset, train or test')
parser.add_argument('--results_folder',
        type=str,
        default='./results/L2_distances',
        help='phase of the dataset, train or test')

args = parser.parse_args()

def L2_distance (keypoints1, keypoints2):
    return sum(sum((keypoints1-keypoints2)**2))

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
    mydict [('-1','-1')]= str(data_root)
    for i in range(0, len(keypoints_paths)-1):
        for j in range(i+1, len(keypoints_paths)):
            print(i,j)
            keypoints1 = np.load(keypoints_paths[i])
            keypoints2 = np.load(keypoints_paths[j])
            distance = L2_distance(keypoints1, keypoints2)
            name_1 = '/'.join(str(keypoints_paths[i]).split('/')[-1:])
            name_1= name_1.split('.')[0]
            name_2 = '/'.join(str(keypoints_paths[j]).split('/')[-1:])
            name_2= name_2.split('.')[0]
            mydict[(name_1,name_2)] = distance
    pickle.dump(mydict,  open(str(save_dir) + "/" + 'L2_distances' + '.pkl', 'wb'))
    print(session, "completed!")

# for key in mydict.keys():
#     print(key)
#     print(mydict[key])


# pickle.dump(mydict,  open(str(save_dir) + "/" + 'L2_distances_'  + str(phase) + '.pkl', 'wb'))

# print(mydict)

# mydict2 = load_pickle('/data4/pantea/nets_implementation/original_bilayer/difficult_poses/L2_distances_train.pkl')
# for key in mydict2.keys():
#     print(key)
#     print(key[0])
#     print(mydict2[key])