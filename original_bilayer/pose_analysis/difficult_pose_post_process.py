import glob
import pathlib
import numpy as np
import pickle
import pdb
import os
import argparse
import shutil 

parser= argparse.ArgumentParser("Difficult pose post process")
parser.add_argument('--pickle_root',
        default= '/data/pantea/pose_results/difficult_poses/per_person_1_three_datasets',
        type=str,
        help='pickle root')
parser.add_argument('--phase',
        type=str,
        default='test',
        help='phase of the dataset, train or test')
parser.add_argument('--min_per_session_keypoints',
        type=int,
        default=2,
        help='each pickle file should contain at least a number of keypoints')

args = parser.parse_args()

def load_pickle(path_string):
    pkl_file = open(path_string, 'rb')
    my_dict = pickle.load(pkl_file)
    pkl_file.close()
    return my_dict

pickle_root = args.pickle_root
phase = args.phase
dataset_name =  str(pickle_root).split('/')[-1:][0]
min_per_session_keypoints = args.min_per_session_keypoints

keypoint_directory=pathlib.Path(pickle_root+'/'+phase)
print(keypoint_directory)
# Find all the video sessions
keypoints_sessions = keypoint_directory.glob('*/*/*')
keypoints_sessions = sorted([str(seq) for seq in keypoints_sessions])

for session in keypoints_sessions:
    my_dict = load_pickle(session +  "/" + 'Difficult_poses' + '.pkl')
    if len(my_dict) < 1+ min_per_session_keypoints:
        print(session, " removed. Not enough difficult poses.")
        shutil.rmtree(pathlib.Path(session)) 
    else:
        print(session, "has enough images.")


# Find all the videos and delete the empty ones
videos = keypoint_directory.glob('*/*')
videos = sorted([str(seq) for seq in videos])
for video in videos:
    #Getting the list of directories
    dir = os.listdir(video)
    # Checking if the list is empty or not
    if len(dir) == 0:
        print("Deleting empty directory of ", video)
        shutil.rmtree(pathlib.Path(video)) 
    else:
        print(video, " is not an empty directory")
