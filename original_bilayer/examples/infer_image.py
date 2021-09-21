"""
Test script meant for a particular pair of two test images, do not use for larger inference tasks, use infer.py instead.
This file, loads two images as target and source image and performs inference on the checkpoints of a trained model. 

Arguments
----------
preprocess : If you want to preprocess two images, put this to True, if not the code will load preprocessed images and keypoints.
draw_source_target_from_video : If preprocess==True, you have the option two choose two frames from a video as source and target frames by setting this flag to True.
experiment_name : The name of the experiment that you want to test
experiment_dir  : The root of experiments
init_which_epoch: The epoch that you want to test

If you set preprocess and draw_source_target_from_video to True, you will use two frames from a video as source and target images. You need to change the following variables:
    video_path : Path to the video (Example: '/video-conf/scratch/pantea/temp_dataset/id00012/_raOc3-IRsw/00110.mp4')
    source_frame_num : The frame number of the source  (Example: 0)
    target_frame_num : The frame number of the target  (Example: 10)

If you set preprocess to True and draw_source_target_from_video to False, you will use to images as source and target imges. You need to define these paths like:
    source_img_path = Full path to the source image (Example: '/home/pantea/NETS/nets_implementation/original_bilayer/examples/images/video_imgs/train_frames/0.jpg')
    target_img_path = Full path to the source image (Example: '/home/pantea/NETS/nets_implementation/original_bilayer/examples/images/video_imgs/train_frames/1.jpg')

If you set preprocess and draw_source_target_from_video to False, you will load the images, keypoints, and segmentations from stored datasets: 
    dataset_root = The dataset root (Example: '/video-conf/scratch/pantea/temp_extracts')
    source_relative_path = The source image's relative path to dataset_root/imgs (Example: 'train/id00012/_raOc3-IRsw/00110/0')
    target_relative_path = The target image's relative path to dataset_root/imgs (Example: 'train/id00012/_raOc3-IRsw/00110/1')


You can set preprocess and draw_source_target_from_video in these orders:
from_video = draw_source_target_from_video

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

The output images are saved with suffix {preprocess}_{draw_source_target_from_video}. 

"""

# Importing libraries
import sys
sys.path.append('../')
import os
import glob
import copy
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
import math
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import cv2
from examples import utils as infer_utils

# Util functions
def make_lut_u():
    return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)

def make_lut_v():
    return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)

def convert_PLI_to_YUV (pli_image, name_to_save):
    pil_image = pli_image.convert('RGB') 
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    img = open_cv_image[:, :, ::-1].copy() 
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)
    # lut_u, lut_v = make_lut_u(), make_lut_v()
    # # Convert back to BGR so we can apply the LUT and stack the images
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    # u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
    # v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    # u_mapped = cv2.LUT(u, lut_u)
    # v_mapped = cv2.LUT(v, lut_v)
    # result = np.vstack([img, y, u_mapped, v_mapped])
    # cv2.imwrite(str(args.save_dir) + '/' + str(name_to_save) + '.png', result)
    return y

# Parser
parser= argparse.ArgumentParser("Inference of models")

parser.add_argument('--experiment_dir',
        type=str,
        default= '/data/pantea/pantea_experiments_chunky/per_person/from_paper',
        help='root directory where the experiment and its checkpoints are saved ')

parser.add_argument('--experiment_name',
        type=str,
        default= 'close_source_target_original_easy_diff_combo',
        help='associated name of the experimnet')

parser.add_argument('--which_epoch',                                     
        type=str,
        default='2000',
        help='epoch to infer from')

parser.add_argument('--video_path',
        type=str,
        default='/video-conf/vedantha/voxceleb2/dev/mp4/id00018/5BVBfpfzjIk/00006.mp4',
        help='path to the video that we collect the two test frames from')

parser.add_argument('--source_frame_num',
        type=int,
        default=0,
        help='frame number of the video in video_path for source')
        
parser.add_argument('--target_frame_num',
        type=int,
        default=1,
        help='frame number of the video in video_path for target')

parser.add_argument('--dataset_root',
        type=str,
        default='/video-conf/scratch/pantea/per_person_1_three_datasets',
        help='root to the dataset')

parser.add_argument('--source_relative_path',
        type=str,
        default='train/id00015/0fijmz4vTVU/00001/0',
        help='realtive path to source image from train/test/etc')

parser.add_argument('--target_relative_path',
        type=str,
        default='train/id00015/0fijmz4vTVU/00001/1',
        help='realtive path to target image from train/test/etc')

parser.add_argument('--source_img_path',
        type=str,
        default='/home/pantea/NETS/nets_implementation/original_bilayer/examples/images/video_imgs/train_frames/0.jpg',
        help='absolute path to source image')

parser.add_argument('--target_img_path',
        type=str,
        default='/home/pantea/NETS/nets_implementation/original_bilayer/examples/images/video_imgs/train_frames/1.jpg',
        help='absolute path to target image')

parser.add_argument('--save_dir',
        type=str,
        default= './results/images',
        help='the directory to save the generated images')       

parser.add_argument('--preprocess',
        type=bool,
        default= False,
        help='If preprocess is needed')   

parser.add_argument('--draw_source_target_from_video',
        type=bool,
        default= False,
        help='If source-target pair is from a video')   

args = parser.parse_args()


# Inputs 
preprocess = args.preprocess
draw_source_target_from_video = args.draw_source_target_from_video

# Checkpoints
experiment_dir = args.experiment_dir
experiment_name = args.experiment_name
which_epoch = args.which_epoch

# Path to the saved dataset when preprocess is False
dataset_root = args.dataset_root
source_relative_path = args.source_relative_path
target_relative_path = args.target_relative_path

# Path to the images when preprocess is True and draw_source_target_from_video is False
source_img_path = args.source_img_path
target_img_path = args.target_img_path

# Video options if both preprocess and draw_source_target_from_video are True
video_path = args.video_path
source_frame_num = int(args.source_frame_num)
target_frame_num = int(args.target_frame_num)

# Instantiate the Inference Module
args_dict = infer_utils.get_model_input_arguments(experiment_dir, experiment_name, which_epoch)
module = InferenceWrapper(args_dict)
input_data_dict = {}
if preprocess and not draw_source_target_from_video:
    input_data_dict = {
        'source_imgs': np.asarray(Image.open(source_img_path)), # H x W x 3
        'target_imgs': np.asarray(Image.open(target_img_path))  # H x W x 3
    }

# Prints for information    
if preprocess:
    if draw_source_target_from_video == False:
        print("You are using input_data_dict of two images that you entered and use preprocess.")
    else:
        print("You are reading a video and pick two frames as target and source images and use preprocess.")
else:
    if draw_source_target_from_video == True:
        print("Can not load images from videos without preprocess. Setting draw_source_target_from_video to False.")
    draw_source_target_from_video = False
    print("You are loading preprocessed .jpg, .npy, .png imags, keypoints, and segmentations.")

# Pass the inputs to the Inference Module
output_data_dict = module(input_data_dict,
                          preprocess= preprocess,
                          draw_source_target_from_video = draw_source_target_from_video,
                          video_path = video_path,
                          source_frame_num=source_frame_num,
                          target_frame_num=target_frame_num,
                          dataset_root = dataset_root,
                          source_relative_path=source_relative_path,
                          target_relative_path=target_relative_path) 

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

rgb_psnr, ssim_value, lpips_value, target, predicted_target = infer_utils.process_output_data_dict(output_data_dict, False)

target_y = convert_PLI_to_YUV (target, "target")
predicted_target_y = convert_PLI_to_YUV (predicted_target, "predicted_target")
target_y_float = np.array(target_y).astype(np.float32)
predicted_target_y_float = np.array(predicted_target_y).astype(np.float32)
yuv_psnr = infer_utils.per_frame_psnr(target_y_float, predicted_target_y_float)
print("rgb_psnr, yuv_psnr, ssim_value, lpips_value", rgb_psnr, yuv_psnr, ssim_value, lpips_value)

# Save the output images
np.save(str(args.save_dir) + '/metrics.npy', np.array([rgb_psnr, yuv_psnr, ssim_value, lpips_value]))
desired_keys = ['pred_target_imgs', 'target_stickmen', 'source_stickmen', 'source_imgs', \
'target_imgs', 'source_segs', 'target_segs', 'pred_target_delta_hf_rgbs', 'pred_tex_hf_rgbs', 'pred_target_uvs', 'pred_target_delta_uvs']


for key in ['pred_target_uvs', 'pred_target_delta_uvs']:
    if key in output_data_dict.keys():
        pred_array = output_data_dict[key].cpu()[0,0].numpy()
        np.save("{}/{}_{}_{}".format(str(args.save_dir), str(key), str(preprocess), str(draw_source_target_from_video)), pred_array)

for key in desired_keys: 
    if key in output_data_dict.keys():
        pred_img = infer_utils.to_image(output_data_dict[key][0, 0])
        pred_img.save("{}/{}_{}_{}.png".format(str(args.save_dir), str(key), str(preprocess), str(draw_source_target_from_video)))

for key, output_seg in zip (['source_imgs', 'target_imgs'], ['source_segs', 'target_segs']):
    if key in output_data_dict.keys():
        masked_source_imgs = infer_utils.to_image(output_data_dict[key] [0, 0], output_data_dict[output_seg] [0, 0])
        masked_source_imgs.save("{}/masked_{}_{}_{}.png".format(str(args.save_dir), str(key), str(preprocess), str(draw_source_target_from_video)))

print("Done!")  
