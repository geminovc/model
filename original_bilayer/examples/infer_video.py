"""
Test script is used for recreating a video with the model. 
The arguments are discussed in infer_image module in detail. 

Different Arguments
--------------------
video_technique: it could be:\
        1) last_frame_next_source: previous video frame is used as source for the current frame \
        2) last_predicted_next_source: previous predicted frame is used as source for the current frame \
        3) representative_sources: for each video frame in each bin, a pre-chosen frame from the same bin is selected as source'

relative_path_base: relative path to images, keypoints, segmentations, and yaws of a session that we want to reproduce.  
yaw_root: The absolute path to the yaw directory 

Outputs
----------


"""

# Importing libraries
import sys
sys.path.append('../')
import os
import glob
import copy
import pdb
import pathlib
import cv2
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import time
from infer import InferenceWrapper
import argparse
from natsort import natsorted
from torchvision import transforms
import argparse
import moviepy.editor as mp
import math
from skimage.metrics import structural_similarity as ssim
import lpips
from examples import utils as infer_utils


# Util functions
def load_session_yaws (yaw_npy_paths):
    yaw_dict = {}
    for yaw_path in yaw_npy_paths:
        frame_num = (str(yaw_path).split('/')[-1]).split('.')[0]
        yaw_dict [frame_num] = load_npy(yaw_path)
    return yaw_dict

def load_npy(path_string):
    np_array = np.load(path_string)
    yaw = np_array [0]
    return yaw

# This function finds all the existing bins in a session. 
# yaw_dict is the yaw dictionary of a session containing yaw_dict[frame_number] = yaw
def find_session_bins (yaw_dict, bins):
    frames_bin_dict = {}
    representatives = {}
    for current_bin in bins:
        frames = [k for k,v in yaw_dict.items() if (v>= current_bin[0] and v<= current_bin[1])]
        if len(frames) >= 2:
            # representatives [bin] = the frame_number that is selected as the representative of the bin
            representatives[str(current_bin)] = frames[0]
            for x in frames:
                # frames_bin_dict [frame_number] = the bin that (frame_number)_th frame belongs to 
                frames_bin_dict[x] = str(current_bin)
    return representatives , frames_bin_dict


# Parser
parser= argparse.ArgumentParser("Video making")

parser.add_argument('--experiment_dir',
        type=str,
        default= '/data/pantea/pantea_experiments_chunky/per_person/from_paper',
        help='root directory where the experiment and its checkpoints are saved ')

parser.add_argument('--experiment_name',
        type=str,
        default= 'original_frozen_Gtex_from_identical',
        help='associated name of the experimnet')

parser.add_argument('--which_epoch',                                     
        type=str,
        default='2000',
        help='epoch to infer from')

parser.add_argument('--video_technique',
        type=str,
        default='last_frame_next_source',
        help='it could be:\
        1) last_frame_next_source: previous video frame is used as source for the current frame \
        2) last_predicted_next_source: previous predicted frame is used as source for the current frame \
        3) representative_sources: for each video frame in each bin, a pre-chosen frame from the same bin is selected as source')

parser.add_argument('--video_path',
        type=str,
        default='/video-conf/vedantha/voxceleb2/dev/mp4/id00015/0fijmz4vTVU/00001.mp4',
        help='path to the video you want to reproduce')

parser.add_argument('--dataset_root',
        type=str,
        default='/video-conf/scratch/pantea/per_person_1_three_datasets',
        help='root to the dataset containg images')

parser.add_argument('--relative_path_base',
        type=str,
        default='train/id00015/0fijmz4vTVU/00001',
        help='realtive path to images from train/test/etc')

parser.add_argument('--yaw_root',
        type=str,
        default= '/video-conf/scratch/pantea/pose_results/yaws/per_person_1_three_datasets/angles',
        help='the directory where the yaws are stored in voxceleb2 format')

parser.add_argument('--save_dir',
        type=str,
        default= './results/videos',
        help='the directory to save the generated video')       


args = parser.parse_args()

# Inputs 
experiment_dir = args.experiment_dir
experiment_name = args.experiment_name
which_epoch = args.which_epoch
video_technique = args.video_technique

# Instantiate the Inference Module
args_dict = infer_utils.get_model_input_arguments(experiment_dir, experiment_name, which_epoch)
module = InferenceWrapper(args_dict)

predicted_video = []
target_video = []
psnr_values = []
ssim_values = []
lpips_values = []
video_self_psnr = []

if video_technique != 'representative_sources':
    video_path = pathlib.Path(args.video_path)
    video = cv2.VideoCapture(str(video_path))
    frame_num = 0
    
    while video.isOpened():
        ret, frame = video.read()
        if frame is None:
            break

        frame = frame[:,:,::-1]
        if frame_num==0:
            source_frame = frame

        target_frame = frame
        if frame_num!=0 and video_technique == 'last_frame_next_source':
            video_self_psnr.append(infer_utils.per_frame_psnr(np.array(source_frame), np.array(target_frame)))
         
        input_data_dict = {
        'source_imgs': np.asarray(np.array(source_frame)), # H x W x 3
        'target_imgs': np.asarray(np.array(target_frame))  # H x W x 3
        }
        # Pass the inputs to the Inference Module
        output_data_dict = module(input_data_dict, preprocess= True, draw_source_target_from_video = False)
        psnr_values, ssim_values, lpips_values, target_video, predicted_video = infer_utils.process_output_data_dict ( 
                                                                                                           output_data_dict,
                                                                                                           True, 
                                                                                                           target_video,
                                                                                                           predicted_video,
                                                                                                           psnr_values,
                                                                                                           ssim_values,
                                                                                                           lpips_values)
        if video_technique == 'last_frame_next_source':
            source_frame = frame
        elif video_technique == 'last_predicted_next_source':
            img_tensor = output_data_dict['pred_target_imgs'][0, 0]
            img_tensor = torch.mul(torch.add(img_tensor, 1), 0.5).clamp(0, 1).permute(1,2,0)
            source_frame = (np.array(img_tensor.cpu())*255).astype(np.uint8)

        print("frame number %d reproduced!" %(frame_num))
        frame_num += 1
        
else:
    # Path to the saved dataset when preprocess is False
    dataset_root = args.dataset_root
    source_relative_path_base = args.relative_path_base
    target_relative_path_base = args.relative_path_base
    bins = [[-90,-80], [-80,-70],[-70,-60],[-60,-50],[-50,-40],[-40,-30],[-30,-20],[-20,-10],\
    [-10,0],[0,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80],[80,90]]
    session =  args.yaw_root + '/' + target_relative_path_base
    yaw_npy_paths = sorted(list(pathlib.Path(str(session)).glob('*')))
    yaw_dict = load_session_yaws(yaw_npy_paths)
    representatives , frames_bin_dict = find_session_bins(yaw_dict, bins)
    print("representatives",representatives)
    keys = sorted(frames_bin_dict.keys())
    
    for i in range(0 , len(yaw_npy_paths)):
        if str(i) in frames_bin_dict.keys():
            print("frame", i , "bin", frames_bin_dict[str(i)])
            target_relative_path = target_relative_path_base +'/'+str(i)
            source_relative_path = source_relative_path_base +'/'+str(representatives[str(frames_bin_dict[str(i)])])
            #Pass the inputs to the Inference Module
            output_data_dict = module(data_dict = {},
                                    preprocess= False,
                                    draw_source_target_from_video = False,
                                    dataset_root = dataset_root,
                                    source_relative_path=source_relative_path,
                                    target_relative_path=target_relative_path)

            psnr_values, ssim_values, lpips_values, target_video, predicted_video = infer_utils.process_output_data_dict (output_data_dict,
                                                                                                            True,
                                                                                                            target_video,
                                                                                                            predicted_video,
                                                                                                            psnr_values,
                                                                                                            ssim_values,
                                                                                                            lpips_values)
# Save the output images
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


print("psnr_values mean", sum(psnr_values)/len(psnr_values))
print("ssim_values mean", sum(ssim_values)/len(ssim_values))
print("lpips_values mean",sum(lpips_values)/len(lpips_values))
np.save(str(args.save_dir) + '/' + video_technique + ".npy", np.array([min(psnr_values),sum(psnr_values)/len(psnr_values),max(psnr_values),
                                                                       min(ssim_values),sum(ssim_values)/len(ssim_values),max(ssim_values),
                                                                       min(lpips_values),sum(lpips_values)/len(lpips_values),max(lpips_values)]))


if video_technique == 'last_frame_next_source':
    print("Video self psnr (frame(n), frame(n-1))")
    print(min(video_self_psnr),sum(video_self_psnr)/len(video_self_psnr),max(video_self_psnr))


# Making videos
# original video and audio
original_video = mp.VideoFileClip(str(pathlib.Path(args.video_path)))
original_audio = original_video.audio
# construct the predicted video
predicted_imgs = [np.array(i) for i in predicted_video]
predicted_clips = [mp.ImageClip(m).set_duration(0.04) for m in predicted_imgs]
concat_clip = mp.concatenate_videoclips(predicted_clips, method="compose")
concat_clip_new = concat_clip.set_audio(original_audio.set_duration(concat_clip.duration))
concat_clip_new.write_videofile(str(args.save_dir) + '/' + video_technique + ".mp4",fps=25)
# construct the masked original video
original_imgs = [np.array(i) for i in target_video]
original_clips = [mp.ImageClip(m).set_duration(0.04) for m in original_imgs]
concat_clip2 = mp.concatenate_videoclips(original_clips, method="compose")
concat_clip2_new = concat_clip2.set_audio(original_audio.set_duration(concat_clip.duration))
concat_clip2_new.write_videofile(str(args.save_dir) + '/masked_original.mp4',fps=25)
# stacking the original and predicted videos
final_stacked = mp.clips_array([[concat_clip_new , concat_clip2]])
final_stacked.write_videofile(str(args.save_dir) + '/' + video_technique + "_stacked" + ".mp4",fps=25)

cv2.destroyAllWindows()
        

print("Done!")  