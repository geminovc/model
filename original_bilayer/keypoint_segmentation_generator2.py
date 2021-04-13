
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


class Keypoint_Segmentation_Generator():
    @staticmethod
    def get_args(parser):
        # Common properties
        parser.add('--data_root',                default="/data/pantea/video_conf", type=str,
                                                 help='root directory of the train data')
        parser.add('--keypoint_dir',                default="keypoint", type=str,
                                                 help='root directory of the stored keypoints')
        parser.add('--video_root',                default="/video-conf/vedantha/voxceleb2/dev/mp4/", type=str,
                                                 help='root directory of the raw videos')        
        parser.add('--segmentatio_dir',                default="segs", type=str,
                                                 help='root directory of the data')                                                                                             
        parser.add('--num_source_frames',     default=1, type=int,
                                              help='number of frames used for initialization of the model')

        parser.add('--num_target_frames',     default=1, type=int,
                                              help='number of frames per identity used for training')

        parser.add('--image_size',            default=256, type=int,
                                              help='output image size in the model')

        parser.add('--num_keypoints',         default=68, type=int,
                                              help='number of keypoints (depends on keypoints detector)')

        parser.add('--output_segmentation',   default='True', type=rn_utils.str2bool, choices=[True, False],
                                              help='read segmentation mask')

        parser.add('--output_stickmen',       default='True', type=rn_utils.str2bool, choices=[True, False],
                                              help='draw stickmen using keypoints')
        
        parser.add('--stickmen_thickness',    default=2, type=int, 
                                              help='thickness of lines in the stickman')

        return parser

    def __init__(self, args, phase):        
        # Store options
        self.phase = phase
        self.args = args


        self.to_tensor = transforms.ToTensor()

        data_root = self.args['data_root']
        
        # Data paths
        self.video_dir = pathlib.Path(elf.args['video_root']) 
        self.imgs_dir = pathlib.Path(data_root) / 'imgs' / phase
        self.pose_dir = pathlib.Path(data_root) / 'keypoints' / phase

        print("imgs_dir", self.imgs_dir)
        print("pose_dir", self.pose_dir)

        # Video sequences list
        sequences = self.imgs_dir.glob('*/*')
        self.sequences = ['/'.join(str(seq).split('/')[-2:]) for seq in sequences]
        
        if args['output_segmentation']:
            self.segs_dir = pathlib.Path(data_root) / 'segs' / phase
        
        self.to_tensor = transforms.ToTensor()

        # Stickman/facemasks drawer
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

        #self.net_seg = wrapper.SegmentationWrapper(self.args)

    
    def change_args(self, args_dict):
        self.args = self.get_args(args_dict)


    def preprocess_data(self, input_imgs, crop_data=True):
        imgs = []
        poses = []
        stickmen = []


        pose = self.fa.get_landmarks(input_imgs)[0]

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
            img = Image.fromarray(input_imgs)

            if crop_data:
                img = img.crop((center[0]-size, center[1]-size, center[0]+size, center[1]+size))
                s = img.size[0]
                pose -= center - size

            img = img.resize((self.args['image_size'], self.args['image_size']), Image.BICUBIC)

            imgs.append((self.to_tensor(img) - 0.5) * 2)

        if crop_data:
            pose = pose / float(s)

        poses.append(np.reshape(((pose - 0.5) * 2),-1))

        if self.args['output_stickmen']:
            stickmen = ds_utils.draw_stickmen(self.args, poses[0])
            stickmen = stickmen[None]

        if input_imgs is not None:
            imgs.append(input_imgs)


            


        segs = None
        if hasattr(self, 'net_seg') and not isinstance(imgs, list):
            segs = self.net_seg(imgs)[None]

        return poses, imgs, segs, stickmen

   



    def read_dataset (self):
         # Sample source and target frames for the current sequence
        
        index=0
        filenames = [1]
        while len(filenames):
            try:
                filenames_img = list((self.imgs_dir / self.sequences[index]).glob('*/*'))
                filenames_img = [pathlib.Path(*filename.parts[-4:]).with_suffix('') for filename in filenames_img]

                filenames = list(set(filenames_img))

                if len(filenames)!=0:
                    break
                else:
                    raise # the length of filenames_img is zero.

            except Exception as e:
                print("# Exception is raised if filenames list is empty or there was an error during read")
                index = (index + 1) % len(self)


        filenames = sorted(filenames)

        imgs = []
        poses = []
        stickmen = []
        segs = []


        for frame_num in range(0, len(filenames)):
            
            filename = filenames[frame_num]
            # Read images
            img_path = pathlib.Path(self.imgs_dir) / filename.with_suffix('.jpg')
            print(img_path)

            #try:
            img = Image.open(img_path)
            img = np.asarray(img).reshape(img.size[0], img.size[1],3)
            img_x = img.copy()
            # Preprocess an image
            img_x = img_x.resize((256,256))
            poses, imgs, segs, stickmen = self.preprocess_data(img, crop_data=True)

            save_name = str(self.pose_dir) + "/" +str(filename) 
            if not os.path.exists(save_name):
                os.makedirs(save_name) 
            np.save(str(self.pose_dir) +"/" +str(filename) , poses)
            print("kkk")
            if self.args['output_segmentation']:
                segs.save(self.segs_dir + filename, segs)
            
            #except Exception as e:
            #    print("# Exception man!")
