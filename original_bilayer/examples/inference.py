import sys
sys.path.append('../')

import glob
import copy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import time
from infer import InferenceWrapper
import argparse


# def get_args(parser=argparse.ArgumentParser()):
#     # General options
#     parser.add('--experiment_dir',          default='.', type=str,
#                                             help='directory to save logs')
    
#     parser.add('--pretrained_weights_dir',  default='/video_conf/scratch/pantea', type=str,
#                                             help='directory for pretrained weights of loss networks (lpips , ...)')
    
#     parser.add('--experiment_dir',            default='.', type=str,
#                                             help='root directory of the experiments')

#     parser.add('--experiment_name',         default='test', type=str,
#                                             help='name of the experiment used for logging')
    
#     # Distributed options
#     parser.add('--num_gpus',                 default=1, type=int,
#                                                 help='>1 enables DDP')

#     # Initialization options
#     parser.add('--init_experiment_dir',     default='', type=str,
#                                             help='directory of the experiment used for the initialization of the networks')

#     parser.add('--init_networks',           default='', type=str,
#                                             help='list of networks to intialize')

#     parser.add('--init_which_epoch',        default='none', type=str,
#                                             help='epoch to initialize from')

#     parser.add('--psnr_loss_apply_to',      default='pred_target_delta_lf_rgbs , target_imgs', type=str,
#                                             help='psnr loss to apply') 

#     return parser

experiment_name = 'all_networks_frozen_except_after_upsampling_blocks_in_texture_generator_and_inference_generator'
experiment_dir = '/video-conf/scratch/pantea_experiments_chunky'
init_which_epoch = '100'
source_path = 'images/20.jpg'
target_path = 'images/109.jpg'


args_dict = {
    'experiment_dir': experiment_dir,
    'pretrained_weights_dir': '/video-conf/scratch/pantea', 
    'init_experiment_dir': project_dir + '/runs/' + experiment_name ,
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': init_which_epoch,
    'num_gpus': 1,
    'experiment_name': experiment_name,
    'psnr_loss_apply_to': 'pred_target_delta_lf_rgbs, target_imgs',
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'inf_apply_masks': False}


module = InferenceWrapper(args_dict)



input_data_dict = {
    'source_imgs': np.asarray(Image.open(source_path)), # H x W x 3
    'target_imgs': np.asarray(Image.open(target_path))[None]} # B x H x W x # 3

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
pred_target.save(str(experiment_name )+ "_" +str(init_which_epoch) + "_" + "pred_target_imgs.jpg")
