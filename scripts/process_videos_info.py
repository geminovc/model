"""
This script records the following information for videos in --video_root:

    'video_path','width','height', 'num_frames','avg_frame_rate', 'bit_rate','duration'

Inputs
----------

video_root: root to the video directory. The videos should be stored in voxceleb2 format.
result_file_name: name of csv file to write results out to


Outputs
----------

The output is a csv file called result_file_name that contains the coulmns: 'video_path','width','height', 'num_frames','avg_frame_rate', 'bit_rate','duration'

Sample usage:

python get_videos_info.py --video_root '/path/to/videos' --result_file_name results.csv


"""

from __future__ import unicode_literals, print_function
import argparse
import ffmpeg
import sys
import glob
import pathlib
import csv
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Get video information')
parser.add_argument('--video_root',
                        type=str,
                        default = '/video-conf/vedantha/voxceleb2/dev/mp4',
                        help='root directory of the videos')

parser.add_argument('--result_file_name',
                       type=str,
                       required=True,
                       help='name of csv file to write results out to')




if __name__ == '__main__':
    args = parser.parse_args()

    data_dict = {}
    cvs_file_name = str(args.result_file_name)    
    with open(cvs_file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            count+=1
            if row['pix_fmt']!= 'yuv420p':
                print(row['video_path'])
            data_dict[row['video_path']] = row['bit_rate']
    print("num videos", count)
    values = [int(v) for v in data_dict.values() ]


    plt.figure()
    plt.hist(values, bins=30)  # density=False would make counts
    plt.savefig('hist.png')
    max_keys = [k for k, v in data_dict.items() if int(v) >= max(values)*2/3 ] # getting all keys containing the `maximum`
    #print(max_keys)