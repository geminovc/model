import sys
sys.path.append('../')
import os
import glob
import copy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import time
from infer_from_paper import InferenceWrapper
import argparse



##Load the model

args_dict = {
    'experiment_dir': '/video-conf/scratch/pantea_experiments_mapmaker',
    'pretrained_weights_dir': '/video-conf/scratch/pantea', 
    'init_experiment_dir': '/video-conf/scratch/pantea_experiments_mapmaker/runs/all_networks_frozen_except_inference_generator',
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'metrics': 'PSNR, lpips',
    'psnr_loss_apply_to': 'pred_target_delta_lf_rgbs, target_imgs',
    'init_which_epoch': '1000',
    'num_gpus': 1,
    'experiment_name': 'all_networks_frozen_except_inference_generator',
    'which_epoch': '1000',
    'stickmen_thickness': 2, 
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'enh_apply_masks': False,
    'inf_apply_masks': False}
    
module = InferenceWrapper(args_dict)

# ## your input frames location here:
# target_folder = natsorted(glob.glob('images/target_video/*.png'))
# target_image_list = []

# for img in target_folder:
#     target_image_list.append(np.asarray(Image.open(img)))


input_data_dict = {
    'source_imgs': np.asarray(Image.open('images/0-1.jpg')), # H x W x 3
    'target_imgs': np.array(Image.open('images/133.jpg')) } # B x H x W x # 3

output_data_dict = module(input_data_dict)
print(output_data_dict['pred_target_imgs'].shape)

def to_image(img_tensor, seg_tensor=None):
    img_array = ((img_tensor.clamp(-1, 1).cpu().numpy() + 1) / 2).transpose(1, 2, 0) * 255
    
    if seg_tensor is not None:
        seg_array = seg_tensor.cpu().numpy().transpose(1, 2, 0)
        img_array = img_array * seg_array + 255. * (1 - seg_array)

    return Image.fromarray(img_array.astype('uint8'))
    
for i in range(1):
    pred_img = to_image(output_data_dict['pred_target_delta_lf_rgbs'][0, i], output_data_dict['target_segs'][0, i])
    # save location
    if not os.path.exists("results/"):
        os.makedirs("results")
    pred_img.save("results/{}.png".format(str(i)))  