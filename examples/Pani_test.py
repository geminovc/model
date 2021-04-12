import sys
sys.path.append('../')

import glob
import copy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import sys
sys.path.append("/home/pantea/video-conf/pantea/bilayer-model")
from infer import InferenceWrapper

args_dict = {
    'project_dir': '../',
    'init_experiment_dir': '../runs/vc2-hq_adrianb_paper_main',
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': '2225',
    'num_gpus': 1,
    'experiment_name': 'vc2-hq_adrianb_paper_enhancer',
    'which_epoch': '1225',
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'enh_apply_masks': False,
    'inf_apply_masks': False}


module = InferenceWrapper(args_dict)
input_data_dict = {
    'source_imgs': np.asarray(Image.open('images/target.jpg')), # H x W x 3
    'target_imgs': np.asarray(Image.open('images/source.jpg'))[None]} # B x H x W x # 3

output_data_dict = module(input_data_dict)

def to_image(img_tensor, seg_tensor=None):
    img_array = ((img_tensor.clamp(-1, 1).cpu().numpy() + 1) / 2).transpose(1, 2, 0) * 255
    
    if seg_tensor is not None:
        seg_array = seg_tensor.cpu().numpy().transpose(1, 2, 0)
        img_array = img_array * seg_array + 255. * (1 - seg_array)

    return Image.fromarray(img_array.astype('uint8'))

source_img = to_image(output_data_dict['source_imgs'][0, 0])
print(source_img)

hf_texture = to_image(output_data_dict['pred_enh_tex_hf_rgbs'][0, 0])
print(hf_texture)

target_pose = to_image(output_data_dict['target_stickmen'][0, 0])
(target_pose)

pred_img = to_image(output_data_dict['pred_enh_target_imgs'][0, 0], output_data_dict['pred_target_segs'][0, 0])
print(pred_img)