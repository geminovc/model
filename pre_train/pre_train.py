import argparse
import glob
import copy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from infer import InferenceWrapper
import argparse
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import sys
import pathlib
import numpy as np
import cv2
import importlib
import ssl

from datasets import utils as ds_utils
from runners import utils as rn_utils
from external.Graphonomy import wrapper
import face_alignment


args_dict = {
    'project_dir': '.',
    'init_experiment_dir': './runs/vc2-hq_adrianb_paper_main',
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': '2225',
    'num_gpus': 1,
    'experiment_name': 'vc2-hq_adrianb_paper_enhancer',
    'which_epoch': '1225',
    'spn_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, texture_enhancer',
    'enh_apply_masks': False,
    'inf_apply_masks': False}

# Initialization
module = InferenceWrapper(args_dict)

# Input data for intiialization and inference
data_dict = {
    'source_imgs': np.asarray(Image.open('/pre_train/images/source.jpg'))  # Size: H x W x 3, type: NumPy RGB uint8 image
    'target_imgs': np.asarray(Image.open('/pre_train/images/target.jpg'))[None] # Size: NUM_FRAMES x H x W x 3, type: NumPy RGB uint8 images
}

# Inference
data_dict = module(data_dict)

# Outputs (images are in [-1, 1] range, segmentation masks -- in [0, 1])
imgs = data_dict['pred_enh_target_imgs']
segs = data_dict['pred_target_segs']

# Stickman/facemasks drawer
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

net_seg = wrapper.SegmentationWrapper(self.args)


def preprocess_data(self, input_imgs, crop_data=True):
        imgs = []
        poses = []
        stickmen = []

        if len(input_imgs.shape) == 3:
            input_imgs = input_imgs[None]
            N = 1

        else:
            N = input_imgs.shape[0]

        for i in range(N):
            pose = self.fa.get_landmarks(input_imgs[i])[0]

            center = ((pose.min(0) + pose.max(0)) / 2).round().astype(int)
            size = int(max(pose[:, 0].max() - pose[:, 0].min(), pose[:, 1].max() - pose[:, 1].min()))
            center[1] -= size // 6

            if input_imgs is None:
                # Crop poses
                if crop_data:
                    s = size * 2
                    pose -= center - size

            else:
                # Crop images and poses
                img = Image.fromarray(input_imgs[i])

                if crop_data:
                    img = img.crop((center[0]-size, center[1]-size, center[0]+size, center[1]+size))
                    s = img.size[0]
                    pose -= center - size

                img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)

                imgs.append((self.to_tensor(img) - 0.5) * 2)

            if crop_data:
                pose = pose / float(s)

            poses.append(torch.from_numpy((pose - 0.5) * 2).view(-1))

        poses = torch.stack(poses, 0)[None]

        if self.args.output_stickmen:
            stickmen = ds_utils.draw_stickmen(self.args, poses[0])
            stickmen = stickmen[None]

        if input_imgs is not None:
            imgs = torch.stack(imgs, 0)[None]

        if self.args.num_gpus > 0:
            poses = poses.cuda()
            
            if input_imgs is not None:
                imgs = imgs.cuda()

                if self.args.output_stickmen:
                    stickmen = stickmen.cuda()

        segs = None
        if hasattr(self, 'net_seg') and not isinstance(imgs, list):
            segs = self.net_seg(imgs)[None]

        return poses, imgs, segs, stickmen
