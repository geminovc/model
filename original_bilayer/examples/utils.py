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

loss_fn = lpips.LPIPS(net='alex')
# transform for centering picture
regular_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# This function transforms Inference output images to savable .jpg
# Inputs:
# img_tensor: the tensor that we want to save
# seg_tensor : optional, the inference segmentation to mask the background
# Outputs:
# PIL Image to save (masked with the seg_tensor optionally)
def to_image(img_tensor, seg_tensor=None):
    if seg_tensor is not None:
        img_tensor = img_tensor * seg_tensor + (-1) * (1 - seg_tensor)
    img_tensor = torch.mul(torch.add(img_tensor, 1), 0.5).clamp(0, 1)
    to_image_module = transforms.ToPILImage()
    img_tensor = img_tensor.cpu()
    return to_image_module(img_tensor)


# Computes the psnr of two input images x, y
def per_frame_psnr(x, y):
    assert(x.size == y.size)
    mse = np.mean(np.square(x - y))
    if mse > 0:
        psnr = 10 * math.log10(255*255/mse)
    else:
        psnr = 100000
    return psnr


# Computes psnr, lpips, and ssim for img1 and img2
def compute_metric_for_files(img1, img2):
    ssim_value = ssim(np.array(img1), np.array(img2), multichannel=True)
    psnr_value = per_frame_psnr(np.array(img1).astype(np.float32), np.array(img2).astype(np.float32)) # float 32 is very important!
    
    img1 = regular_transform(Image.fromarray(np.array(img1)))
    img2 = regular_transform(Image.fromarray(np.array(img2)))
    lpips_value = torch.flatten(loss_fn(img1, img2).detach())[0].item()
    return psnr_value, ssim_value, lpips_value


# Process the output of the model, which is in the form of dictionary, to get the metrics and final images/videos 
def process_output_data_dict (output_data_dict, if_make_video, target_video = [], predicted_video = [],
 psnr_values = [], ssim_values = [], lpips_values = []):
    predicted_target = to_image(output_data_dict['pred_target_imgs'][0, 0], output_data_dict['target_segs'] [0, 0])
    target = to_image(output_data_dict['target_imgs'][0, 0], output_data_dict['target_segs'] [0, 0])
    psnr_frame, ssim_frame , lpips_frame = compute_metric_for_files(target, predicted_target)
    if if_make_video:
        predicted_video.append(predicted_target)
        target_video.append(target)
        psnr_values.append(psnr_frame)
        ssim_values.append(ssim_frame)
        lpips_values.append(lpips_frame)
        return  psnr_values, ssim_values, lpips_values, target_video, predicted_video
    else:
        return  psnr_frame, ssim_frame , lpips_frame, target, predicted_target


# Assigning correct argument dictionary and input data dictionary
def get_model_input_arguments (experiment_dir, experiment_name, which_epoch):
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
    'replace_source_specific_with_trainable_tensors': False,
    'dropout_networks': 'texture_generator: 0.5',
    'use_dropout': False,
    'texture_output_dim': 3,
    'use_unet': False,
    'time_networks': False,
    'replace_Gtex_output_with_source': False}
    
    return args_dict