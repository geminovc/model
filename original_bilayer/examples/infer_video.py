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
import argparse
import moviepy.editor as mp
import math
from skimage.metrics import structural_similarity as ssim
import lpips
import torch

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
        help='root to the dataset')

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

loss_fn = lpips.LPIPS(net='alex')
# transform for centering picture
regular_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

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
    if seg_tensor is not None:
        img_tensor = img_tensor * seg_tensor + (-1) * (1 - seg_tensor)
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
    frames_bin_dict = {}
    representatives = {}
    for current_bin in bins:
        frames = [k for k,v in yaw_dict.items() if (v>= current_bin[0] and v<= current_bin[1])]
        if len(frames) >= 2:
            representatives[str(current_bin)] = frames [0]
            for x in frames:
                frames_bin_dict[x] = str(current_bin)
    return representatives , frames_bin_dict

def per_frame_psnr(x, y):
    assert(x.size == y.size)

    mse = np.mean(np.square(x - y))
    if mse > 0:
        psnr = 10 * math.log10(255*255/mse)
    else:
        psnr = 100000
    return psnr

""" compute metric for all filenames matching prefix relative to the reference video """
def compute_metric_for_files(img1, img2):
    # read one frame at a time and compute average metric

    ssim_value = ssim(np.array(img1), np.array(img2), multichannel=True)
    psnr_value = per_frame_psnr(np.array(img1).astype(np.float32), np.array(img2).astype(np.float32)) # float 32 is very important!
    img1 = regular_transform(Image.fromarray(np.array(img1)))
    img2 = regular_transform(Image.fromarray(np.array(img2)))
    
    lpips_value = torch.flatten(loss_fn(img1, img2).detach())[0].item()
    
    return psnr_value, ssim_value, lpips_value

def process_output_data_dict (output_data_dict, target_video, predicted_video, psnr_values, ssim_values, lpips_values):
    predicted_target = to_image(output_data_dict['pred_target_imgs'][0, 0], output_data_dict['target_segs'] [0, 0])
    predicted_video.append(predicted_target)
    target = to_image(output_data_dict['target_imgs'][0, 0], output_data_dict['target_segs'] [0, 0])
    target_video.append(target)
    psnr_frame, ssim_frame , lpips_frame = compute_metric_for_files(target, predicted_target)
    psnr_values.append(psnr_frame)
    ssim_values.append(ssim_frame)
    lpips_values.append(lpips_frame)
    return target_video, predicted_video, psnr_values, ssim_values, lpips_values 
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
    'replace_source_specific_with_trainable_tensors': False,
    'dropout_networks': 'texture_generator: 0.5',
    'use_dropout': False,
    'texture_output_dim': 3,
    'use_unet': False }


# Instantiate the Inference Module
module = InferenceWrapper(args_dict)

predicted_video = []
target_video = []
psnr_values = []
ssim_values = []
lpips_values = []
video_self_psnr = []

if video_technique != 'representative_sources':
    # Opening the video
    video_path = pathlib.Path(args.video_path)
    # '/video-conf/vedantha/voxceleb2/dev/mp4/id00015/1mPH6AESHus/00021.mp4'
    video = cv2.VideoCapture(str(video_path))
    frame_num = 0

    # Reading the video frame by frame
    while video.isOpened():
        ret, frame = video.read()
        if frame is None:
            break
        
        frame = frame[:,:,::-1]
        if  frame_num==0:
            source_frame = frame
        
        target_frame = frame
        
        if frame_num!=0 and video_technique == 'last_frame_next_source':
            video_self_psnr.append(per_frame_psnr(np.array(source_frame), np.array(target_frame)))
        
        
        input_data_dict = {
        'source_imgs': np.asarray(np.array(source_frame)), # H x W x 3
        'target_imgs': np.asarray(np.array(target_frame))  # H x W x 3
        }

        # Pass the inputs to the Inference Module
        output_data_dict = module(input_data_dict, preprocess= True, draw_source_target_from_video = False)
        target_video, predicted_video, psnr_values, ssim_values, lpips_values = process_output_data_dict ( output_data_dict,
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
        frame_num+=1
        


else:
    # Path to the saved dataset when preprocess is False
    dataset_root = args.dataset_root
    source_relative_path_base = args.relative_path_base
    target_relative_path_base = args.relative_path_base
    bins = [[-90,-80], [-80,-70],[-70,-60],[-60,-50],[-50,-40],[-40,-30],[-30,-20],[-20,-10],[-10,0],[0,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80],[80,90]]
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

            target_video, predicted_video, psnr_values, ssim_values, lpips_values = process_output_data_dict (output_data_dict,
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
# original_audio.write_audiofile("audio.mp3")

# constructing the predicted video
imgs = [np.array(i) for i in predicted_video]
clips = [mp.ImageClip(m).set_duration(0.04) for m in imgs]
concat_clip = mp.concatenate_videoclips(clips, method="compose")
concat_clip_new = concat_clip.set_audio(original_audio.set_duration(concat_clip.duration))
concat_clip_new.write_videofile(str(args.save_dir) + '/' + video_technique + ".mp4",fps=25)

# txt_clip = mp.TextClip("predicted", fontsize = 75, color = 'white') 
# txt_clip = txt_clip.set_pos('center').set_duration(concat_clip.duration) 
# concat_clip_new = CompositeVideoClip([concat_clip_new, txt_clip]) 

# construct the masked original video
imgs2 = [np.array(i) for i in target_video]
clips2 = [mp.ImageClip(m).set_duration(0.04) for m in imgs2]
concat_clip2 = mp.concatenate_videoclips(clips2, method="compose")
concat_clip2_new = concat_clip2.set_audio(original_audio.set_duration(concat_clip.duration))
concat_clip2_new.write_videofile(str(args.save_dir) + '/masked_original.mp4',fps=25)

# stacking the original and predicted videos
# final = mp.clips_array([[concat_clip_new], [original_video]])
final_stacked = mp.clips_array([[concat_clip_new , concat_clip2]])
final_stacked.write_videofile(str(args.save_dir) + '/' + video_technique + "_stacked" + ".mp4",fps=25)



cv2.destroyAllWindows()
        

print("Done!")  