"""
model = BilayerAPI(config_path)
source_pose = model.extract_keypoints(source_frame, 'source')
target_pose = model.extract_keypoints(target_frame, 'target')
predicted_target = model.predict(target_pose, target_segs)

"""


# Loading libraries
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
import pdb
import time
from datasets import utils as ds_utils
from runners import utils as rn_utils
from external.Graphonomy import wrapper
import face_alignment
import yaml

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class BilayerAPI(nn.Module):

    def type_fix(self, args_dict):
        for key in args_dict.keys():
            value = args_dict[key]
            value, _ = rn_utils.typecast_value(key, str(value))
            print(key, value)
        return args_dict
        

    def convert_yaml_to_dict(self, config_path):
        with open(config_path) as f:
            args_dict = yaml.safe_load(f)

        # args_dict = self.type_fix(args_dict)
        args_dict = AttrDict(args_dict)
        return args_dict

    def __init__(self, config_path):
        super(BilayerAPI, self).__init__()
        # Get a config for the network
        self.args = self.convert_yaml_to_dict(str(config_path))
        print("self.args", self.args)
        self.to_tensor = transforms.ToTensor()

        # Load the runner file and set it to the evaluation(test) mode
        self.runner = importlib.import_module(f'runners.{self.args.runner_name}').RunnerWrapper(self.args, training=False)
        self.runner.eval()

        # Load checkpoints from experiment
        checkpoints_dir = pathlib.Path(self.args.experiment_dir) / 'runs' / self.args.experiment_name / 'checkpoints'

        # Load pre-trained weights
        init_networks = rn_utils.parse_str_to_list(self.args.init_networks) if self.args.init_networks else {}
        networks_to_train = self.runner.nets_names_to_train
        print("init_networks:", init_networks)

        # Initialize the model with experiment weights
        if self.args.init_which_epoch != 'none' and self.args.init_experiment_dir:
            for net_name in init_networks:
                print("loaded ", net_name, "from ", str(pathlib.Path(self.args.init_experiment_dir) / 'checkpoints' / f'{self.args.init_which_epoch}_{net_name}.pth'))
                self.runner.nets[net_name].load_state_dict(torch.load(pathlib.Path(self.args.init_experiment_dir) / 'checkpoints' / f'{self.args.init_which_epoch}_{net_name}.pth', map_location='cpu'))

        for net_name in networks_to_train:
            if net_name not in init_networks and net_name in self.runner.nets.keys():
                print("loaded ", net_name, "from ", str(checkpoints_dir / f'{self.args.which_epoch}_{net_name}.pth'))
                self.runner.nets[net_name].load_state_dict(torch.load(checkpoints_dir / f'{self.args.which_epoch}_{net_name}.pth', map_location='cpu'))
        
        # Remove spectral norm to improve the performance
        self.runner.apply(rn_utils.remove_spectral_norm)

        # Stickman/facemasks drawer
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
        
        # Segmentation Wrapper module
        self.net_seg = wrapper.SegmentationWrapper(self.args)

        # Initialize an empty dictionary
        self.data_dict = {}

        if self.args.num_gpus > 0:
            self.cuda()

    def change_args(self, args_dict):
        self.args = self.get_args(args_dict)

    # Set the source-target imgs, poses, segs in self.data_dict
    def extract_keypoints(self, frame, image_name='target'):

        poses, imgs, segs, stickmen = self.preprocess_data(frame, crop_data=True)
        self.data_dict['{}_imgs'.format(str(image_name))] = imgs
        self.data_dict['{}_poses'.format(str(image_name))] = poses

        if segs is not None:
            self.data_dict['{}_segs'.format(str(image_name))] = segs

        if stickmen is not None:
            self.data_dict['{}_stickmen'.format(str(image_name))] = stickmen
        return poses
    
    def preprocess_data(self, input_imgs, crop_data=True):
        imgs = []
        poses = []
        stickmen = []        
        
        # Finding the batch-size of the input imgs
        if len(input_imgs.shape) == 3:
            input_imgs = input_imgs[None]
            N = 1

        else:
            N = input_imgs.shape[0]

        # Iterate over all the images in the batch
        for i in range(N):

            # Get the pose of the i-th image in the batch 
            pose = self.fa.get_landmarks(input_imgs[i])[0]

            # Finding the center of the face using the pose coordinates
            center = ((pose.min(0) + pose.max(0)) / 2).round().astype(int)

            # Finding the maximum between the width and height of the image 
            size = int(max(pose[:, 0].max() - pose[:, 0].min(), pose[:, 1].max() - pose[:, 1].min()))
            center[1] -= size // 6

            img = Image.fromarray(input_imgs[i])

            if crop_data:
                # Crop images and poses
                img = img.crop((center[0]-size, center[1]-size, center[0]+size, center[1]+size))
                s = img.size[0]
                pose -= center - size

            img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
            
            # This step is not done in keypoints_segmentations_extraction module
            imgs.append((self.to_tensor(img) - 0.5) * 2)
        
            # This step is not done in keypoints_segmentations_extraction module
            if crop_data:
                pose = pose / float(s)
            
            # This step is not done in keypoints_segmentations_extraction module
            poses.append(torch.from_numpy((pose - 0.5) * 2).view(-1))
        
        # Stack the poses from different images
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
        
        # Get the segmentations
        segs = None
        if hasattr(self, 'net_seg') and not isinstance(imgs, list):
            segs = self.net_seg(imgs)[None]

        return poses, imgs, segs, stickmen


    def predict(self, target_poses, target_segs=None):

        # Setting the model in test mode
        model = self.runner
        model.eval()

        # Prepare input data
        if self.args.num_gpus > 0:
            for key, value in data_dict.items():
                self.data_dict[key] = value.cuda()

        # Forward pass
        with torch.no_grad():
            model(self.data_dict)

        return model.data_dict['predicted_target_imgs']
