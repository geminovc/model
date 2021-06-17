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
        default='./results',
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