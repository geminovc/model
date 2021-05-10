
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
        self.video_dir = pathlib.Path("/video-conf/vedantha/voxceleb2/dev/mp4/") #'/video-conf/scratch/pantea/0
        self.imgs_dir = pathlib.Path(data_root) / 'imgs' / phase
        self.pose_dir = pathlib.Path(data_root) / 'keypoints' / phase

        print("imgs_dir", self.imgs_dir)
        print("pose_dir", self.pose_dir)

        # Video sequences list
        sequences = self.video_dir.glob('*/*')
        self.sequences = ['/'.join(str(seq).split('/')[-2:]) for seq in sequences]
        
        if args['output_segmentation']:
            self.segs_dir = pathlib.Path(data_root) / 'segs' / phase
        
        self.to_tensor = transforms.ToTensor()

        # Stickman/facemasks drawer
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

        #self.net_seg = wrapper.SegmentationWrapper(self.args)


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
            #if hasattr(self, 'net_seg') and not isinstance(imgs, list):
            #    segs = self.net_seg(imgs)[None]

            return poses, imgs, segs, stickmen

    def get_poses (self):
         # Sample source and target frames for the current sequence
        for index in range(0,len(self.sequences)):
            print("Sequences is: ", self.sequences[index])
            filenames_vid = list((self.video_dir / self.sequences[index]).glob('*'))
            filenames_vid = [pathlib.Path(*filename.parts[-3:]).with_suffix('') for filename in filenames_vid]
            filenames = list(set(filenames_vid))
            filenames = sorted(filenames)
            for filename in filenames:
                video_path = pathlib.Path(self.video_dir) / filename.with_suffix('.mp4')
                name = str(filename).split('/')[len(str(filename).split('/'))-1]                                
                video = cv2.VideoCapture(str(video_path))
                frame_num = 0
                offset = 0 
                while video.isOpened():
                    ret, frame = video.read()
                    if frame is None:
                        break
                    if offset > 0:
                        offset-= 1
                        continue
                    if frame_num % 30 == 0:
                        print("frame number is: ",frame_num)

                    frame = frame[:,:,::-1]
                    try: 
                        poses, imgs, segs, stickmen = self.preprocess_data(frame, crop_data=True)
                        if poses is not None and len(poses) == 1:
                            imgs_directory = str(self.imgs_dir) +"/"+ str(filename) + "/" + str(frame_num)
                            keypoints_directory = str(self.pose_dir) +"/"+ str(filename) + "/" + str(frame_num)
                            os.makedirs(str(self.imgs_dir) +"/"+ str(filename), exist_ok=True)
                            os.makedirs(str(self.pose_dir) +"/"+ str(filename), exist_ok=True)
                            img = Image.fromarray(frame)
                            img.save(imgs_directory + str(index) + '.jpg')
                            np.save(keypoints_directory, poses)
                            #os.makedirs(segs_directory, exist_ok=True)

                    except: 
                        print("Excaption happened in reading the poses of the frame.") 

                    frame_num+=1
                video.release()
                #imgs_directory = 
              
