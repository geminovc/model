"""
This script exposes endpoints to the Bilayer Inference pipeline

1.  To initialte the model use:
    model = BilayerAPI(config_path)

    where the config_path is the path to the config yaml file

2.  To set the target and source images, use the following APIs:
    source_pose, _ = model.extract_keypoints(source_frame, 'source')
    target_pose, target_pose = model.extract_keypoints(target_frame, 'target')

    Note that each time you want to update the source or target frame, just call these APIs.

3.  To find the predicted PLI image based source images, use the following API:
    predicted_target = model.predict(target_pose, target_segs)

Example:

# Paths
config_path = '/path/to/yaml_file'
source_img_path = '/path/to/source_image'
target_img_path = '/path/to/target_image'

# Convert to numpy
source_frame = np.asarray(Image.open(source_img_path))
target_frame = np.asarray(Image.open(target_img_path))

model = BilayerAPI(config_path)

source_poses = model.extract_keypoints(source_frame)
target_poses = model.extract_keypoints(target_frame)

model.update_source(source_poses, source_frame)

# Passing the Target Frame
predicted_target = model.predict(target_poses, target_frame)
predicted_target.save("pred_target_with_the_target_frame.png") #TODO expose APIs for metrics and stickmen

# Not Passing the Target Frame
predicted_target = model.predict(target_poses)
predicted_target.save("pred_target_without_the_target_frame.png")

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
from examples import utils as infer_utils
from external.Graphonomy import wrapper
import face_alignment
import yaml

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class BilayerAPI(nn.Module):
    def __init__(self, config_path):
        super(BilayerAPI, self).__init__()
        # Get a config for the network
        self.args = self.convert_yaml_to_dict(str(config_path))
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

        if self.args.num_gpus > 0:
            # Stickman/facemasks drawer
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
        else:
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, \
                                        flip_input=True, device='cpu')
        # Segmentation Wrapper module
        self.net_seg = wrapper.SegmentationWrapper(self.args)

        # Initialize an empty dictionary
        self.data_dict = {}

        if self.args.num_gpus > 0:
            self.cuda()

    # Fix the type for non-str
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

    # Get the pose of the input frame
    def extract_keypoints(self, frame):
        pose = self.fa.get_landmarks(frame)[0]
        return pose

    # Get the stickmen for the pose
    def get_stickmen(self, poses):
        stickmen = ds_utils.draw_stickmen(self.args, poses)
        stickmen = stickmen[None]
        return stickmen

    # Get the segmentations
    def get_segmentations(self, imgs):
        segs = None
        if imgs is not None:
            if hasattr(self, 'net_seg') and not isinstance(imgs, list):
                segs = self.net_seg(imgs)[None]

        return segs

    def normalize_frame_and_poses(self, pose, input_frame, crop_data=True):
        imgs = []
        poses = []

        if input_frame is not None:
            # Adding a new dimention for consistency
            if len(input_frame.shape) == 3:
                input_frame = input_frame[None]

        # Finding the center of the face using the pose coordinates
        center = ((pose.min(0) + pose.max(0)) / 2).round().astype(int)

        # Finding the maximum between the width and height of the image
        size = int(max(pose[:, 0].max() - pose[:, 0].min(), pose[:, 1].max() - pose[:, 1].min()))
        center[1] -= size // 6

        if input_frame is not None:
            img = Image.fromarray(input_frame[0])
        else:
            img = None

        if crop_data:
            # Crop images and poses
            if img is not None:
                img = img.crop((center[0]-size, center[1]-size, center[0]+size, center[1]+size))
                self.s = img.size[0]
            pose -= center - size

        if img is not None:
            img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
            # This step is not done in keypoints_segmentations_extraction module
            imgs.append((self.to_tensor(img) - 0.5) * 2)
        else:
            imgs = None

        # This step is not done in keypoints_segmentations_extraction module
        if crop_data:
            pose = pose / float(self.s)
        
        # This step is not done in keypoints_segmentations_extraction module
        poses.append(torch.from_numpy((pose - 0.5) * 2).view(-1))
        poses = torch.stack(poses, 0)[None]

        if input_frame is not None:
            imgs = torch.stack(imgs, 0)[None]

        return poses, imgs

    # Scale and crop and resize frame and poses
    # Find Stickmen and segmentations
    def preprocess_data(self, pose, input_frame, image_name, crop_data=True):
        stickmen = []        

        poses, imgs = self.normalize_frame_and_poses(pose, input_frame, crop_data)

        if self.args.output_stickmen:
            stickmen = self.get_stickmen(poses[0])

        # Get the segmentations
        segs = self.get_segmentations(imgs)

        if self.args.num_gpus > 0:
            poses, imgs, stickmen = assign_to_cuda(poses, imgs, stickmen)

        self.update_data_dict(imgs, poses, segs, stickmen, image_name)

    # Updates the values in the data_dict
    def update_data_dict(self, imgs, poses, segs, stickmen, image_name):
        self.data_dict['{}_imgs'.format(str(image_name))] = imgs
        self.data_dict['{}_poses'.format(str(image_name))] = poses
        self.data_dict['{}_segs'.format(str(image_name))] = segs
        self.data_dict['{}_stickmen'.format(str(image_name))] = stickmen

    # Assign the variables to cuda if using gpu
    def assign_to_cuda(self, poses, imgs, stickmen):
        poses = poses.cuda()
        if imgs is not None:
            imgs = imgs.cuda()

            if self.args.output_stickmen:
                stickmen = stickmen.cuda()

        return poses, imgs, stickmen


    # Updates the source frame for inference
    def update_source(self, source_poses, source_frame):
        print("Updated the source frame")
        # Set the variables of data_dict for source image
        self.preprocess_data(source_poses, source_frame, 'source', crop_data=True)

    # Predicts an image based on the target_pose and the source_frame
    # source_frame has been set in update_source fucntion
    def predict(self, target_poses, target_frame=None):
        # Set the variables of data_dict for target image
        self.preprocess_data(target_poses, target_frame, 'target', crop_data=True)
        # Setting the model in test mode
        model = self.runner
        model.eval()

        # Prepare input data
        if self.args.num_gpus > 0:
            for key, value in self.data_dict.items():
                self.data_dict[key] = value.cuda()

        # Forward pass
        with torch.no_grad():
            model(self.data_dict)

        predicted_tensor = model.data_dict['pred_target_imgs'][0, 0]
        predicted_pli_img = infer_utils.to_image(predicted_tensor)
        print("Prediction successfully finished!")
        return predicted_pli_img
