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


# Set the arguments 
preprocess = True

if preprocess:
    print("You are using input_data_dict of two images that you entered and use preprocess.")
else:
    print("You are loading preprocessed .jpg, .npy, .png imags, keypoints, and segmentations.")

args_dict = {
    'experiment_dir': '/video-conf/scratch/pantea_experiments_chunky', #directory that the experiment is saved in
    'experiment_name': 'new_keypoints_per_person_with_augmentation_by_general_from_base_lessen_general_data_loader',
    'which_epoch': '1800',
    'init_experiment_dir': '/video-conf/scratch/pantea_experiments_chunky/runs/new_keypoints_per_person_with_augmentation_by_general_from_base_lessen_general_data_loader',
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': '1800',
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
    
module = InferenceWrapper(args_dict)

# ## your input frames location here:
# target_folder = natsorted(glob.glob('images/*.jpg'))
# target_image_list = []

# for img in target_folder:
#     target_image_list.append(np.asarray(Image.open(img)))


input_data_dict = {
    'source_imgs': np.asarray(Image.open('/home/pantea/NETS/nets_implementation/original_bilayer/examples/images/video_imgs/train_frames/0.jpg')), # H x W x 3
    'target_imgs': np.asarray(Image.open('/home/pantea/NETS/nets_implementation/original_bilayer/examples/images/video_imgs/train_frames/1.jpg')) # H x W x 3
}


# 'target_imgs': np.array(target_image_list) } # B x H x W x # 3

from_video = True
output_data_dict = module(input_data_dict,  preprocess= preprocess, from_video = from_video) 

def to_image(img_tensor, seg_tensor=None):
    img_tensor = torch.mul(torch.add(img_tensor, 1), 0.5).clamp(0, 1)
    to_image_module = transforms.ToPILImage()
    img_tensor = img_tensor.cpu()
    return to_image_module(img_tensor)

#for i in range(len(target_image_list)):
for i in range(1):
    
    # Save location
    if not os.path.exists("results/"):
        os.makedirs("results")
    
    pred_img = to_image(output_data_dict['pred_target_imgs'][0, i], output_data_dict['target_segs'][0, i])
    pred_img.save("results/pred_target_{}_{}_{}.jpg".format(str(i), str(preprocess), str(from_video)))  

    if 'target_stickmen' in output_data_dict.keys():
        target_stickmen = to_image(output_data_dict['target_stickmen'][0, i])
        target_stickmen.save("results/target_stickmen_{}_{}_{}.png".format(str(i), str(preprocess), str(from_video)))

    if 'source_stickmen' in output_data_dict.keys():
        source_stickmen = to_image(output_data_dict['source_stickmen'][0, i])
        source_stickmen.save("results/source_stickmen_{}_{}_{}.png".format(str(i), str(preprocess), str(from_video)))
    
    if 'source_imgs' in output_data_dict.keys():
        source_imgs = to_image(output_data_dict['source_imgs'][0, i])
        source_imgs.save("results/source_imgs_{}_{}_{}.png".format(str(i), str(preprocess), str(from_video)))

    if 'target_imgs' in output_data_dict.keys():
        target_imgs = to_image(output_data_dict['target_imgs'][0, i])
        target_imgs.save("results/target_imgs_{}_{}_{}.png".format(str(i), str(preprocess), str(from_video)))

    if 'source_segs' in output_data_dict.keys():
        source_segs = to_image(output_data_dict['source_segs'][0, i])
        source_segs.save("results/source_segs_{}_{}_{}.png".format(str(i), str(preprocess), str(from_video)))

    if 'target_segs' in output_data_dict.keys():
        target_segs = to_image(output_data_dict['target_segs'][0, i])
        target_segs.save("results/target_segs_{}_{}_{}.png".format(str(i), str(preprocess), str(from_video)))


print("Done!")  