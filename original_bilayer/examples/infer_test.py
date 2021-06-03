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
preprocess = True
from_video = False
experiment_dir = '/video-conf/scratch/pantea_experiments_chunky'
experiment_name = 'new_keypoints_per_person_with_augmentation_by_general_from_base_lessen_general_data_loader'
which_epoch = '1800'

# Path to the saved dataset when preprocess is False
dataset_root = '/video-conf/scratch/pantea/temp_extracts'
source_relative_path = 'train/id00012/_raOc3-IRsw/00110/0'
target_relative_path = 'train/id00012/_raOc3-IRsw/00110/1'

# Path to the images when preprocess is True and from_video is False
source_img_path = '/home/pantea/NETS/nets_implementation/original_bilayer/examples/images/video_imgs/train_frames/0.jpg'
target_img_path = '/home/pantea/NETS/nets_implementation/original_bilayer/examples/images/video_imgs/train_frames/1.jpg'


# Video options if both preprocess and from_video are True
video_path = '/video-conf/scratch/pantea/temp_dataset/id00012/_raOc3-IRsw/00110.mp4'
source_frame_num=0
target_frame_num=1


# ------------------------------------------------------------------------------------------------------------------------

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
    'replace_Gtex_output_with_trainable_tensor': False}

input_data_dict = {
    'source_imgs': np.asarray(Image.open(source_img_path)), # H x W x 3
    'target_imgs': np.asarray(Image.open(target_img_path))  # H x W x 3
}

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


# Prints for information    
if preprocess:
    if from_video == False:
        print("You are using input_data_dict of two images that you entered and use preprocess.")
    else:
        print("You are reading a video and pick two frames as target and source images and use preprocess.")
else:
    if from_video == True:
        print("Can not load images from videos without preprocess. Setting from_video to False.")
    from_video = False
    print("You are loading preprocessed .jpg, .npy, .png imags, keypoints, and segmentations.")


# Instantiate the Inference Module
module = InferenceWrapper(args_dict)

# Pass the inputs to the Inference Module
output_data_dict = module(input_data_dict,
                          preprocess= preprocess,
                          from_video = from_video,
                          video_path = video_path,
                          source_frame_num=source_frame_num,
                          target_frame_num=target_frame_num,
                          dataset_root = dataset_root,
                          source_relative_path=source_relative_path,
                          terget_relative_path=target_relative_path) 
    
# Save the output images
if not os.path.exists("results/"):
    os.makedirs("results")

if 'pred_target_imgs' in output_data_dict.keys():
    pred_img = to_image(output_data_dict['pred_target_imgs'][0, 0])
    pred_img.save("results/pred_target_{}_{}.jpg".format(str(preprocess), str(from_video)))  

if 'target_stickmen' in output_data_dict.keys():
    target_stickmen = to_image(output_data_dict['target_stickmen'][0, 0])
    target_stickmen.save("results/target_stickmen_{}_{}.png".format(str(preprocess), str(from_video)))

if 'source_stickmen' in output_data_dict.keys():
    source_stickmen = to_image(output_data_dict['source_stickmen'] [0, 0])
    source_stickmen.save("results/source_stickmen_{}_{}.png".format(str(preprocess), str(from_video)))

if 'source_imgs' in output_data_dict.keys():
    source_imgs = to_image(output_data_dict['source_imgs'] [0, 0])
    source_imgs.save("results/source_imgs_{}_{}.png".format(str(preprocess), str(from_video)))

if 'target_imgs' in output_data_dict.keys():
    target_imgs = to_image(output_data_dict['target_imgs'] [0, 0])
    target_imgs.save("results/target_imgs_{}_{}.png".format(str(preprocess), str(from_video)))

if 'source_segs' in output_data_dict.keys():
    source_segs = to_image(output_data_dict['source_segs'] [0, 0])
    source_segs.save("results/source_segs_{}_{}.png".format(str(preprocess), str(from_video)))

if 'target_segs' in output_data_dict.keys():
    target_segs = to_image(output_data_dict['target_segs'] [0, 0])
    target_segs.save("results/target_segs_{}_{}.png".format(str(preprocess), str(from_video)))


print("Done!")  