'''
This script finds the average psnr of two frame with frame numbers `n` and `n + shift` for all possible combinations
of n and shift in a video and draws the psnr vs shift graph.

Sample usage:

python compute_self_psnr.py --video_path <YOUR_VIDEO_PATH> --save_dir <YOUR_SAVE_DIR> 

'''
# Importing libraries
import os
import pathlib
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
from natsort import natsorted
import argparse
import math


parser= argparse.ArgumentParser("Video self psnr")

parser.add_argument('--video_path',
        type=str,
        default='/video-conf/vedantha/voxceleb2/dev/mp4/id00015/0fijmz4vTVU/00001.mp4',
        help='path to the video')

parser.add_argument('--save_dir',
        type=str,
        default= './results/self_psnr/',
        help='the directory to save the generated graphs')       

args = parser.parse_args()

def per_frame_psnr(x, y):
    assert(x.size == y.size)

    mse = np.mean(np.square(x - y))
    if mse > 0:
        psnr = 10 * math.log10(255*255/mse)
    else:
        psnr = 100000
    return psnr


frames = []
shifts = []
min_psnrs = []
mean_psnrs = []
max_psnrs = []

# Reading the video frame by frame
video_path = pathlib.Path(args.video_path)
video = cv2.VideoCapture(str(video_path))
while video.isOpened():
    ret, frame = video.read()
    if frame is None:
        break
    frame = frame[:,:,::-1]
    frames.append(frame)
print("all frames retrieved")

for shift in range(1, len(frames) - 2):
    shifted_psnr = []
    for frame_num in range(0, len(frames) - shift - 2):
        source_frame = frames [frame_num]
        target_frame = frames [frame_num + shift]
        curr_psnr = per_frame_psnr(np.array(source_frame).astype(np.float32), np.array(target_frame).astype(np.float32))
        shifted_psnr.append(curr_psnr)
    shifts.append(shift)
    min_psnrs.append(min(shifted_psnr))
    print("shift", shift, "avergae psnr in the video",sum(shifted_psnr)/len(shifted_psnr))
    mean_psnrs.append(sum(shifted_psnr)/len(shifted_psnr))
    max_psnrs.append(max(shifted_psnr))

# Save the output images
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Simple mean plot
fig = plt.figure()
ax = plt.axes()
ax.plot(shifts, mean_psnrs)
plt.xlabel ('shift')
plt.ylabel ('psnr')
plt.title('psnr(frame(x), frame(x+shift))')
plt.savefig(str(args.save_dir + 'mean_psnr.png'))

# Plot with error bars
plt.figure()
plt.errorbar(shifts, mean_psnrs, yerr=[min_psnrs , max_psnrs])
plt.xlabel ('shift')
plt.ylabel ('psnr')
plt.title('psnr(frame(x), frame(x+shift))')
plt.savefig(str(args.save_dir + 'video_psnr_with_err.png'))


