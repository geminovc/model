# not complete, change to delete the video when all the sessions become empty
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
        default= '/data4/pantea/nets_implementation/original_bilayer/difficult_poses/results/difficult_poses/random_sampled_per_person',
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

