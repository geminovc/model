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
from infer_from_paper import InferenceWrapper
import argparse
from natsort import natsorted
from torchvision import transforms


##Load the model

args_dict = {
    'croped_segmentation': True, # put this to True
    'experiment_dir': '/video-conf/scratch/pantea',
    'experiment_name': 'vc2-hq_adrianb_paper_main',
    'which_epoch': '2225',
    'init_experiment_dir': '/video-conf/scratch/pantea/bilayer_paper_runs/vc2-hq_adrianb_paper_main',
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': '2225',
    'metrics': 'PSNR, lpips',
    'psnr_loss_apply_to': 'pred_target_delta_lf_rgbs, target_imgs',
    'num_gpus': 1,
    'stickmen_thickness': 2,
    'pretrained_weights_dir': '/video-conf/scratch/pantea', 
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'enh_apply_masks': False,
    'inf_apply_masks': True,
    'dataset_load_from_txt': False}
    
module = InferenceWrapper(args_dict)

# ## your input frames location here:
# target_folder = natsorted(glob.glob('images/*.jpg'))
# target_image_list = []

# for img in target_folder:
#     target_image_list.append(np.asarray(Image.open(img)))


input_data_dict = {
    'source_imgs': np.asarray(Image.open('/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/yi_qz725MjE/00163/1.jpg')), # H x W x 3
    'target_imgs': np.asarray(Image.open('/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/yi_qz725MjE/00163/66.jpg')) # H x W x 3
}
input_data_dict = {
    'source_imgs': np.asarray(Image.open('images/video_imgs/frames/out-015.jpg')), # H x W x 3
    'target_imgs': np.asarray(Image.open('images/video_imgs/frames/out-025.jpg')) # H x W x 3
}
input_data_dict = {
    'source_imgs': np.asarray(Image.open('/video-conf/scratch/pantea/video_conf_datasets/general_dataset/imgs/train/id00012/21Uxsk56VDQ/00001/0.jpg')), # H x W x 3
    'target_imgs': np.asarray(Image.open('/video-conf/scratch/pantea/video_conf_datasets/general_dataset/imgs/train/id00012/21Uxsk56VDQ/00001/100.jpg')) # H x W x 3
}

input_data_dict = {
    'source_imgs': np.asarray(Image.open('/video-conf/scratch/pantea/temp_extracts/imgs/train/id00012/f0Zj2NjXeyE/00139/0.jpg')), # H x W x 3
    'target_imgs': np.asarray(Image.open('/video-conf/scratch/pantea/temp_extracts/imgs/train/id00012/f0Zj2NjXeyE/00139/1.jpg')) # H x W x 3
}


# 'target_imgs': np.array(target_image_list) } # B x H x W x # 3

# If you want to use input_data_dict of two images above and use preprocess, set preprocess to True.
# If you want to load preprocessed .jpg, .npy, .png imags, keypoints, and segmentations, set preprocess to False.
preprocess = False

if preprocess:
    print("You are using input_data_dict of two images that you entered and use preprocess.")
else:
    print("You are loading preprocessed .jpg, .npy, .png imags, keypoints, and segmentations.")


output_data_dict = module(input_data_dict,  preprocess= preprocess) 

def to_image(img_tensor, seg_tensor=None):
    img_tensor = torch.mul(torch.add(img_tensor, 1), 0.5).clamp(0, 1)
    to_image_module = transforms.ToPILImage()
    img_tensor = img_tensor.cpu()
    return to_image_module(img_tensor)

    #img_array = ((img_tensor.clamp(-1, 1).cpu().detach().numpy()  + 1) * 0.5).transpose(1, 2, 0) * 255
    
    # if seg_tensor is not None:
    #     seg_array = seg_tensor.cpu().detach().numpy().transpose(1, 2, 0)
    #     img_array = img_array * seg_array + 255. * (1 - seg_array)

    # return Image.fromarray(img_array.astype('uint8'))
    
#for i in range(len(target_image_list)):
for i in range(1):
    # save location
    if not os.path.exists("results/"):
        os.makedirs("results")
    #pred_img.save("results/pred_target_{}.png".format(str(i)))

    pred_img = to_image(output_data_dict['pred_target_imgs'][0, i], output_data_dict['target_segs'][0, i])
    pred_img.save("results/pred_target_{}.jpg".format(str(i)))  

    if 'target_stickmen' in output_data_dict.keys():
        target_stickmen = to_image(output_data_dict['target_stickmen'][0, i])
        target_stickmen.save("results/target_stickmen_{}.png".format(str(i)))  