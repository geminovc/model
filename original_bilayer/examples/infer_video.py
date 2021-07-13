"""
Test script meant for a particular pair of two test images, do not use for larger inference tasks, use infer.py instead.
This file, loads two images as target and source image and performs inference on the checkpoints of a trained model. 

Arguments
----------
preprocess : If you want to preprocess two images, put this to True, if not the code will load preprocessed images and keypoints.
from_video : If preprocess==True, you have the option two choose two frames from a video as source and target frames by setting this flag to True.
experiment_name : The name of the experiment that you want to test
experiment_dir  : The root of experiments
init_which_epoch: The epoch that you want to test

If you set preprocess and from_video to True, you will use two frames from a video as source and target images. You need to change the following variables:
    video_path : Path to the video (Example: '/video-conf/scratch/pantea/temp_dataset/id00012/_raOc3-IRsw/00110.mp4')
    source_frame_num : The frame number of the source  (Example: 0)
    target_frame_num : The frame number of the target  (Example: 10)

If you set preprocess to True and from_video to False, you will use to images as source and target imges. You need to define these paths like:
    source_img_path = Full path to the source image (Example: '/home/pantea/NETS/nets_implementation/original_bilayer/examples/images/video_imgs/train_frames/0.jpg')
    target_img_path = Full path to the source image (Example: '/home/pantea/NETS/nets_implementation/original_bilayer/examples/images/video_imgs/train_frames/1.jpg')

If you set preprocess and from_video to False, you will load the images, keypoints, and segmentations from stored datasets: 
    dataset_root = The dataset root (Example: '/video-conf/scratch/pantea/temp_extracts')
    source_relative_path = The source image's relative path to dataset_root/imgs (Example: 'train/id00012/_raOc3-IRsw/00110/0')
    target_relative_path = The target image's relative path to dataset_root/imgs (Example: 'train/id00012/_raOc3-IRsw/00110/1')


You can set preprocess and from_video in these orders:

+------------+------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess | from_video |                                                        Source & Target                                                                       |
+============+============+==============================================================================================================================================+
|            |    True    |  Picks two frames (source_frame_num and target_frame_num) from video in video_path preprocess them to find the keypoints                     |
|   True     |============+==============================================================================================================================================+
|            |   False    |  Picks the images in source_img_path and target_img_path and preprocess them to find the keypoints                                           |
+------------+------------+----------------------------------------------------------------------------------------------------------------------------------------------+
|            |    True    |  Not applicable                                                                                                                              |
|   False    |============+==============================================================================================================================================+
|            |   False    |  Loads preprocessed and save keypoints, images, and segmentations from dataset_root/[imgs, keypoints, segs]/{source or target}_relative_path |
+------------+------------+----------------------------------------------------------------------------------------------------------------------------------------------+



Outputs
----------

The following images will be saved in the "results" directory:
'pred_target_imgs'
'target_stickmen'  
'source_stickmen'  
'source_imgs'  
'target_imgs'  
'source_segs'  
'target_segs'  

The output images are saved with suffix {preprocess}_{from_video}. 

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


# Inputs 
experiment_dir = '/data/pantea/pantea_experiments_chunky/per_person/from_paper'
experiment_name = 'close_source_target_original_easy_diff_combo_inf_pred_source_data_True'
which_epoch = '2000'

video_technique = 'representative_sources'
# 'last_predicted_next_source', 'last_frame_next_source'
# ------------------------------------------------------------------------------------------------------------------------
# Util functions

def to_image(img_tensor, seg_tensor=None):
    """ This function transforms Inference output images to savable .jpg
    
    Inputs
    ----------
    img_tensor: the tensor that we want to save
    seg_tensor : optional, the inference segmentation to mask the background

    Returns
    -------
    PIL Image to save (masked with the seg_tensor optionally)

    """
    img_tensor = torch.mul(torch.add(img_tensor, 1), 0.5).clamp(0, 1)
    to_image_module = transforms.ToPILImage()
    img_tensor = img_tensor.cpu()
    return to_image_module(img_tensor)

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

def find_session_bins (yaw_dict, bins):
    bin_dict = {}
    frames_bin_dict = {}
    representatives = {}
    for current_bin in bins:
        frames = [k for k,v in yaw_dict.items() if (v>= current_bin[0] and v<= current_bin[1])]
        if len(frames) >= 2:
            bin_dict[str(current_bin)] = frames
            representatives[str(current_bin)] = frames [0]
            for x in frames:
                frames_bin_dict[x] = str(current_bin)
    return representatives , frames_bin_dict

# Assigning correct argument dictionary and input data dictionary
args_dict = {
    'experiment_dir': experiment_dir,
    'experiment_name': experiment_name,
    'which_epoch': which_epoch,
    'init_experiment_dir': experiment_dir + '/runs/' + experiment_name,
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': which_epoch,
    'metrics': 'PSNR, lpips',
    'psnr_loss_apply_to': 'pred_target_delta_lf_rgbs, target_imgs',
    'num_gpus': 1,
    'stickmen_thickness': 2,
    'pretrained_weights_dir': '/video-conf/scratch/pantea', 
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'enh_apply_masks': False,
    'inf_apply_masks': True,
    'dataset_load_from_txt': False,
    'replace_Gtex_output_with_trainable_tensor': False,
    'replace_source_specific_with_trainable_tensors': False}


# Instantiate the Inference Module
module = InferenceWrapper(args_dict)

if video_technique != 'representative_sources':
    # Opening the video
    video_path = pathlib.Path('/video-conf/vedantha/voxceleb2/dev/mp4/id00018/5BVBfpfzjIk/00006.mp4')
    # '/video-conf/vedantha/voxceleb2/dev/mp4/id00015/1mPH6AESHus/00021.mp4'
    video = cv2.VideoCapture(str(video_path))
    frame_num = 0
    predicted_video = []
    # pdb.set_trace()
    # Reading the video frame by frame
    while video.isOpened():
        print(frame_num)
        ret, frame = video.read()
        if frame is None:
            break
        frame = frame[:,:,::-1]
        if  frame_num==0:
            source_frame = frame
        else:
            target_frame = frame
        
        if frame_num != 0:
            input_data_dict = {
            'source_imgs': np.asarray(np.array(source_frame)), # H x W x 3
            'target_imgs': np.asarray(np.array(target_frame))  # H x W x 3
            }

            # Pass the inputs to the Inference Module
            output_data_dict = module(input_data_dict,preprocess= True, from_video = False)
            predicted_target = to_image(output_data_dict['pred_target_imgs'][0, 0],output_data_dict['target_segs'] [0, 0])
            predicted_video.append(predicted_target)
            if video_technique == 'last_frame_next_source':
                source_frame = frame
            elif video_technique == 'last_predicted_next_source':
                img_tensor = output_data_dict['pred_target_imgs'][0, 0]
                img_tensor = torch.mul(torch.add(img_tensor, 1), 0.5).clamp(0, 1).permute(1,2,0)
                source_frame = (np.array(img_tensor.cpu())*255).astype(np.uint8)
        frame_num+=1

else:
    # Path to the saved dataset when preprocess is False
    dataset_root = '/video-conf/scratch/pantea/per_person_1_three_datasets'
    source_relative_path_base = 'train/id00015/0fijmz4vTVU/00001'
    target_relative_path_base = 'train/id00015/0fijmz4vTVU/00001'
    bins = [[-90,-80], [-80,-70],[-70,-60],[-60,-50],[-50,-40],[-40,-30],[-30,-20],[-20,-10],[-10,0],[0,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80],[80,90]]
    session =  '/video-conf/scratch/pantea/pose_results/yaws/per_person_1_three_datasets/angles/train/id00015/0fijmz4vTVU/00001'

    yaw_npy_paths = sorted(list(pathlib.Path(str(session)).glob('*')))
    yaw_dict = load_session_yaws(yaw_npy_paths)
    representatives , frames_bin_dict = find_session_bins(yaw_dict, bins)
    print("representatives",representatives)
    predicted_video = []
    keys = sorted(frames_bin_dict.keys())
    for i in range(0 , len(yaw_npy_paths)):
        if str(i) in frames_bin_dict.keys():
            print("frame", i , "bin", frames_bin_dict[str(i)])
            target_relative_path = target_relative_path_base +'/'+str(i)
            source_relative_path = source_relative_path_base +'/'+str(representatives[str(frames_bin_dict[str(i)])])
            #Pass the inputs to the Inference Module
            output_data_dict = module(data_dict = {},
                                    preprocess= False,
                                    from_video = False,
                                    dataset_root = dataset_root,
                                    source_relative_path=source_relative_path,
                                    target_relative_path=target_relative_path)
            predicted_target = to_image(output_data_dict['pred_target_imgs'][0, 0],output_data_dict['target_segs'] [0, 0])
            predicted_video.append(predicted_target)
        else:
            print(i, "where are you???")

# Save the output images
if not os.path.exists("results/"):
    os.makedirs("results")

size = (256,256)
video_writer = cv2.VideoWriter("results/"+video_technique+".avi", 0, 25, size)

for video_frame in predicted_video:
    img_rgb = cv2.cvtColor(np.array(video_frame), cv2.COLOR_BGR2RGB)
    video_writer.write(img_rgb)

cv2.destroyAllWindows()
video_writer.release()
        

print("Done!")  