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
#pantea_experiments_mapmaker
#/video-conf/scratch/pantea/bilayer_paper_runs/vc2-hq_adrianb_paper_main
# for the paper's result put croped_segmentation to True
#/video-conf/scratch/pantea_experiments_mapmaker/runs/all_networks_frozen_except_inference_generator
#/video-conf/scratch/pantea_experiments_chunky/runs/embedding_networks_frozen
# /video-conf/scratch/pantea_experiments_mapmaker/runs/all_networks_frozen_except_inference_generator/checkpoints/1000_inference_generator.pth
args_dict = {
    'croped_segmentation': True, # put this to True
    'experiment_dir': '/video-conf/scratch/pantea_experiments_mapmaker',
    'experiment_name': 'all_networks_frozen_except_inference_generator',
    'which_epoch': '1200',
    'init_experiment_dir': '/video-conf/scratch/pantea_experiments_mapmaker/runs/all_networks_frozen_except_inference_generator',
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': '1200',
    'metrics': 'PSNR, lpips',
    'psnr_loss_apply_to': 'pred_target_delta_lf_rgbs, target_imgs',
    'num_gpus': 1,
    'stickmen_thickness': 2,
    'pretrained_weights_dir': '/video-conf/scratch/pantea', 
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'enh_apply_masks': False,
    'inf_apply_masks': False,
    'dataset_load_from_txt': False}
    
module = InferenceWrapper(args_dict)

# ## your input frames location here:
# target_folder = natsorted(glob.glob('images/*.jpg'))
# target_image_list = []

# for img in target_folder:
#     target_image_list.append(np.asarray(Image.open(img)))

# selected test image path is:  /video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/yi_qz725MjE/00163/1.jpg
# selected test image path is:  /video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/yi_qz725MjE/00163/66.jpg
# selected test image path is:  /video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/Z-G8-wqpxwU/00096/74.jpg
# selected test image path is:  /video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/Z-G8-wqpxwU/00105/26.jpg
input_data_dict = {
    'source_imgs': np.asarray(Image.open('/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/yi_qz725MjE/00163/1.jpg')), # H x W x 3
    'target_imgs': np.asarray(Image.open('/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/yi_qz725MjE/00163/66.jpg')) # H x W x 3
}
# 'target_imgs': np.array(target_image_list) } # B x H x W x # 3

print("# Make sure to set crop_data to True or you get terrible results!")
output_data_dict = module(input_data_dict,  no_grad = True, crop_data=True, preprocess= False) # I set this to False and I got terrible results!
print(output_data_dict['pred_target_imgs'].shape)

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

    pred_img = to_image(output_data_dict['pred_target_imgs'][0, i], output_data_dict['pred_target_segs'][0, i])
    # save location
    if not os.path.exists("results/"):
        os.makedirs("results")
    #pred_img.save("results/pred_target_{}.png".format(str(i)))

    pred_lf_img = to_image(output_data_dict['pred_target_delta_lf_rgbs'][0, i], output_data_dict['target_segs'][0, i])
    pred_lf_img.save("results/pred_lf_target_{}.jpg".format(str(i)))  

    # if 'source_stickmen' in output_data_dict.keys():
    #     pred_lf_img = to_image(output_data_dict['source_stickmen'][0, i])
    #     pred_lf_img.save("results/source_stickmen_{}.png".format(str(i)))  