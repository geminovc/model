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
sys.path.append('..')

""" This script exposes endpoints to the Bilayer pipeline

    Example usage (given source and target paths)
    =============================================

    source_frame = np.asarray(Image.open('/path/to/source'))
    target_frame = np.asarray(Image.open('/path/to/target'))

    model = BilayerModel('/path/to/yaml_file')
    source_keypoints = model.extract_keypoints(source_frame)
    target_keypoints = model.extract_keypoints(target_frame)
    model.update_source(source_frame, source_keypoints)
    predicted_target = model.predict(target_keypoints)
    predicted_target.save("pred_target.png")
"""
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class BilayerModel(nn.Module):
    def __init__(self, config_path):
        super(BilayerModel, self).__init__()

        # Get a config for the network
        self.args = self.convert_yaml_to_dict(str(config_path))
        self.to_tensor = transforms.ToTensor()
        self.data_dict = {}

        # Load pre-trained weights
        init_networks = rn_utils.parse_str_to_list(self.args.init_networks) if self.args.init_networks else {}
        print("init_networks:", init_networks)

        # Initialize the model with experiment weights in evaluation(test) mode
        self.runner = importlib.import_module(\
        f'runners.{self.args.runner_name}').RunnerWrapper(self.args, training=False)
        self.runner.eval()

        if self.args.init_which_epoch != 'none' and self.args.init_experiment_dir:
            for net_name in init_networks:
                self.runner.nets[net_name].load_state_dict(torch.load(pathlib.Path(\
                self.args.init_experiment_dir) / 'checkpoints' / f'{self.args.init_which_epoch}_{net_name}.pth'\
                , map_location='cpu'))

        # Remove spectral norm to improve the performance
        self.runner.apply(rn_utils.remove_spectral_norm)

        # Stickman/facemasks drawer
        if self.args.num_gpus > 0:
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, \
                                                    flip_input=True)
        else:
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, \
                                                    flip_input=True, device='cpu')
        # Segmentation Wrapper module
        self.net_seg = wrapper.SegmentationWrapper(self.args)

        if self.args.num_gpus > 0:
            self.cuda()


    def type_fix(self, args_dict):
        """ Typecast the non-str values of the dictionary to
            corresponding type (int, float, etc) """
        for key in args_dict.keys():
            value = args_dict[key]
            value, _ = rn_utils.typecast_value(key, str(value))
        return args_dict


    def convert_yaml_to_dict(self, config_path):
        with open(config_path) as f:
            args_dict = yaml.safe_load(f)
        args_dict = AttrDict(args_dict)
        return args_dict


    def extract_keypoints(self, frame):
        """ extract keypoint from the provided RGB image """
        pose = self.fa.get_landmarks(frame)[0]
        return pose


    def get_stickmen(self, poses):
        stickmen = ds_utils.draw_stickmen(self.args, poses)
        stickmen = stickmen[None]
        return stickmen


    def get_segmentations(self, imgs):
        segs = None
        if imgs is not None:
            if hasattr(self, 'net_seg') and not isinstance(imgs, list):
                segs = self.net_seg(imgs)[None]

        return segs


    def normalize_frame_and_poses(self, pose, input_frame, crop_data=True):
        """ Crops the numpy array RGB input_frame with respect to the top/bottom face positions,
            and normalizes the pose and input_frame to [-1,1]
        """
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

        # Crop images and poses
        if crop_data:
            if img is not None:
                img = img.crop((center[0] - size, center[1] - size, center[0] + size, center[1] + size))
                self.s = img.size[0]
            pose -= center - size
            pose = pose / float(self.s)

        poses.append(torch.from_numpy((pose - 0.5) * 2).view(-1))
        poses = torch.stack(poses, 0)[None]

        if img is not None:
            img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
            imgs.append((self.to_tensor(img) - 0.5) * 2)
            imgs = torch.stack(imgs, 0)[None]
        else:
            imgs = None

        return poses, imgs


    def preprocess_data(self, pose, input_frame, image_name, crop_data=True):
        """ Crop and resize frame and poses, find stickmen and segmentations """
        stickmen = []        

        poses, imgs = self.normalize_frame_and_poses(pose, input_frame, crop_data)

        if self.args.output_stickmen:
            stickmen = self.get_stickmen(poses[0])

        # Get the segmentations
        segs = self.get_segmentations(imgs)

        if self.args.num_gpus > 0:
            poses, imgs, stickmen = self.assign_to_cuda(poses, imgs, stickmen)

        self.update_data_dict(imgs, poses, segs, stickmen, image_name)


    def update_data_dict(self, imgs, poses, segs, stickmen, image_name):
        self.data_dict['{}_imgs'.format(str(image_name))] = imgs
        self.data_dict['{}_poses'.format(str(image_name))] = poses
        self.data_dict['{}_segs'.format(str(image_name))] = segs
        self.data_dict['{}_stickmen'.format(str(image_name))] = stickmen


    def assign_to_cuda(self, poses, imgs, stickmen):
        poses = poses.cuda()
        if imgs is not None:
            imgs = imgs.cuda()

            if self.args.output_stickmen:
                stickmen = stickmen.cuda()

        return poses, imgs, stickmen


    def update_source(self, source_frame, source_keypoints):
        """ update the source and keypoints the frame is using
            from the RGB source provided as input
        """
        print("Updated the source frame")
        # Set the variables of data_dict for source image
        self.preprocess_data(source_keypoints, source_frame, 'source', crop_data=True)


    def predict(self, target_keypoints):
        """ takes target keypoints and returns an RGB image for the prediction """
        # Set the variables of data_dict for target image
        self.preprocess_data(target_keypoints, None, 'target', crop_data=True)

        model = self.runner
        model.eval()

        # Prepare input data
        if self.args.num_gpus > 0:
            for key, value in self.data_dict.items():
                if value is not None:
                    self.data_dict[key] = value.cuda()

        # Forward pass
        with torch.no_grad():
            model(self.data_dict)

        predicted_tensor = model.data_dict['pred_target_imgs'][0, 0]
        predicted_pli_img = infer_utils.to_image(predicted_tensor)
        print("Prediction successfully finished!")
        return predicted_pli_img
