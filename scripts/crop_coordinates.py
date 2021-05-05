""" takes one speaker, iterates through all of the videos in the temporally cropped folder
    and spatially crops them
"""

import sys
import os
import numpy as np 
import argparse
import pickle

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data-dir', metavar='d', type=str,
                    help='path to the folder with all the celeb data', required=True)
parser.add_argument('--pickle-name', metavar='p', type=str,
                    help='name of the pickle file with the coordinates', required=True)
args = parser.parse_args()

input_folder = args.data_dir + "/cropped"
output_folder = args.data_dir + "/spatially_cropped"
pickle_name = args.pickle_name

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


# retrieve annotations
annotations = {}
try:
    with open(pickle_name, "rb") as handle:
        annotations = pickle.load(handle)
except IOError:
    print("Pickle file doesn't exist, returning")
    sys.exit()


# run ffmpeg on each individual input video until all have been spatiall cropped
input_videos = sorted(os.listdir(input_folder))
for input_video in input_videos:
    # extract base name by removing extension
    base_name = input_video.split('.')[0]

    # remove speaker name from the start of the file name
    video_name = base_name[base_name.find('_') + 1:]

    print(input_video, base_name, video_name)

    if os.path.exists(output_folder + "/" + input_video):
        print("Skipping", input_video)
        continue

    # compute coordinates
    if video_name + '.jpg' not in annotations:
        continue
    primary_diag = annotations[video_name + '.jpg']
    top_left_x = primary_diag[0]
    top_left_y = primary_diag[1]
    height = primary_diag[3] - primary_diag[1]
    width = primary_diag[2] - primary_diag[0]
    coordinates = [width, height, top_left_x, top_left_y]
    
    # run ffmpeg
    ffmpeg_command = 'ffmpeg -y -i ' + input_folder + "/" + input_video + \
            ' -filter:v crop="' + ':'.join([str(x) for x in coordinates]) + \
        '" -c:a copy ' + output_folder + "/" + input_video
    os.system(ffmpeg_command)
