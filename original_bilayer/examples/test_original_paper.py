import sys
sys.path.append('../')

import glob
import copy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import time
from infer import InferenceWrapper


#/video-conf/scratch/pantea_experiments/runs/from_paper_pretrained_inf_apply_mask_false_skip_test_true_28_Apr

experiment_name = 'from_paper_pretrained_inf_apply_mask_false_skip_test_true_28_Apr'
project_dir = '/video-conf/scratch/pantea_experiments'

# experiment_name = 'from_base_inf_apply_mask_false_skip_test_false_28_Apr'
# project_dir = '/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/results'

#experiment_name = 'from_base_apply_mask_false'
#project_dir = '/data/pantea/video_conf/one_person_dataset/per_video_dataset/results'


args_dict = {
    'project_dir': project_dir,
    'pretrained_weights_dir': '/video-conf/scratch/pantea', 
    'init_experiment_dir': project_dir + '/runs/' + experiment_name ,
    'init_networks': 'identity_embedder, keypoints_embedder, inference_generator',
    'init_which_epoch': '100',
    'num_gpus': 1,
    'experiment_name': experiment_name,
    'which_epoch': '100',
    'metrics': 'PSNR, lpips',
    'psnr_loss_apply_to': 'pred_target_delta_lf_rgbs, target_imgs',
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'enh_apply_masks': False,
    'inf_apply_masks': False}


module = InferenceWrapper(args_dict)



input_data_dict = {
    'source_imgs': np.asarray(Image.open('images/0-1.jpg')), # H x W x 3
    'target_imgs': np.asarray(Image.open('images/0.jpg'))[None]} # B x H x W x # 3

now = time.time()
output_data_dict = module(input_data_dict)
now_now = time.time()

print("It took ", str(now_now- now), "for load and inference module.")



def to_image(img_tensor, seg_tensor=None):
    img_array = ((img_tensor.clamp(-1, 1).cpu().numpy() + 1) / 2).transpose(1, 2, 0) * 255
    
    if seg_tensor is not None:
        seg_array = seg_tensor.cpu().numpy().transpose(1, 2, 0)
        img_array = img_array * seg_array + 255. * (1 - seg_array)

    return Image.fromarray(img_array.astype('uint8'))


pred_target = to_image(output_data_dict['pred_target_imgs'][0, 0])
pred_target.save("pred_target_imgs.jpg")

# pred_target = to_image(output_data_dict['pred_enh_target_imgs'][0, 0])
# pred_target.save("pred_enh_target_imgs.jpg") 

#pred_segs = to_image(output_data_dict['pred_target_segs'][0, 0])
#pred_segs.save("pred_target_segs.jpg")