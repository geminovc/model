""" gets total duration and number of videos info for a set of specified
    celebrities 
    example invocation:
    python3 get_info.py --data-dir /data/vibhaa/speech2gesture-master/dataset \
            --celebs angelica almaram seth oliver ellen chemistry conan rock shelly
"""

import subprocess
from tabulate import tabulate
import sys
import os
import numpy as np 
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data-dir', metavar='d', type=str,
                    help='path to the folder with all the celeb data', required=True)
parser.add_argument('--celebs', metavar='c', type=str, nargs='+',
                    help='list of celebrities')
args = parser.parse_args()

def get_length(filename):
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                  "format=duration", "-of",
                                 "default=noprint_wrappers=1:nokey=1", filename],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        return float(result.stdout)


def get_video_duration_info(celeb_name):
    total_duration = 0

    input_folder = args.data_dir + "/" + celeb_name + "/spatially_cropped"
    videos = sorted(os.listdir(input_folder))
    num_videos = len(videos)

    for video in videos:
        total_duration += get_length(input_folder + "/" + video)

    print(celeb_name, total_duration, num_videos)
    return total_duration, num_videos


def main():
    info = []
    for celeb in args.celebs:
        length, num_videos = get_video_duration_info(celeb)
        info.append([celeb, length, num_videos])

    print(tabulate(info, headers=["Name", "Video Duration", "Num Videos"]))

main()
