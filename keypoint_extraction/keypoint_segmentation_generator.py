
import argparse
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import sys
sys.path.append("/home/pantea/video-conf/pantea/bilayer-model")
import pathlib
import numpy as np
import cv2
import importlib
import ssl
import time
from datasets import utils as ds_utils
from runners import utils as rn_utils
from external.Graphonomy import wrapper
import face_alignment
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Dataset


class Keypoint_Segmentation_Generator(nn.Module):
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
        super(Keypoint_Segmentation_Generator, self).__init__()        
        # Store options
        self.phase = phase
        self.args = args
        self.frame_count = 0


        self.to_tensor = transforms.ToTensor()

        data_root = self.args['data_root']
        
        # Data paths
        self.video_dir = pathlib.Path("/video-conf/vedantha/voxceleb2/dev/mp4/") #'/video-conf/scratch/pantea/0 #/video-conf/vedantha/voxceleb2/dev/mp4/
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

        if self.args['num_gpus']>0:
            self.cuda()

        #self.dataset = pytorch.dataset_method(self.vid_generator)

        #self.net_seg = wrapper.SegmentationWrapper(self.args)




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
                img = Image.fromarray(np.array(input_imgs[i]))

                if crop_data:
                    img = img.crop((center[0]-size, center[1]-size, center[0]+size, center[1]+size))
                    s = img.size[0]
                    pose -= center - size

                img = img.resize((self.args['image_size'], self.args['image_size']), Image.BICUBIC)

                imgs.append((self.to_tensor(img) - 0.5) * 2)

            if crop_data:
                pose = pose / float(s)

            poses.append(torch.from_numpy((pose - 0.5) * 2).view(-1))

        poses = torch.stack(poses, 0)[None]

        if self.args['output_stickmen']:
            stickmen = ds_utils.draw_stickmen(self.args, poses[0])
            stickmen = stickmen[None]

        if input_imgs is not None:
            imgs = torch.stack(imgs, 0)[None]

        if self.args['num_gpus'] > 0:
            poses = poses.cuda()
            
            if input_imgs is not None:
                imgs = imgs.cuda()

                if self.args['output_stickmen']:
                    stickmen = stickmen.cuda()

        segs = None
        #if hasattr(self, 'net_seg') and not isinstance(imgs, list):
        #    segs = self.net_seg(imgs)[None]

        return poses, imgs, segs, stickmen

    def get_poses (self):
        dataset = Dataset(self.args['sampling_rate'], self.video_dir, self.sequences)
        dataloader = DataLoader(dataset, self.args['batch_size'], shuffle=False)
        self.seq_id = 0
        while int(self.seq_id)<len(self.sequences):
            try:
                self.start = time.time()
                output = next(iter(dataloader))
                frames , frame_nums, filenames, seq_ids = output[0],output[1],output[2], output[3]
                self.seq_id = seq_ids.max()
                #self.frame_count+=1
                print("Learning rate is (frame/sec):", self.args['batch_size']/(time.time()-self.start))
                #
                try: 
                    poses, imgs, segs, stickmen = self.preprocess_data(frames, crop_data=True)

                    ## RESIZE BEFORE STORING!!!!
                    ## MAKE THE SEGMENTATION!!
                    poses = np.array(poses.cpu())
                    frames = (frames.cpu()).numpy()
                    number_of_poses = poses.shape[1]
                    for i in range(0, number_of_poses):
                        pose = poses[:,i,:]
                        frame = frames[i,:,:,:]
                        if pose is not None:
                            imgs_directory = str(self.imgs_dir) +"/"+ str(filenames[i]) + "/" + str(int(frame_nums[i]))
                            keypoints_directory = str(self.pose_dir) +"/"+ str(filenames[i]) + "/" + str(int(frame_nums[i]))
                            os.makedirs(str(self.imgs_dir) +"/"+ str(filenames[i]), exist_ok=True)
                            os.makedirs(str(self.pose_dir) +"/"+ str(filenames[i]), exist_ok=True)
                            frame = frame[:,:,::-1]
                            img = Image.fromarray(frame)
                            img.save(imgs_directory + '.jpg')
                            np.save(keypoints_directory,pose)
                            print("saved to:", imgs_directory)
                            #os.makedirs(segs_directory, exist_ok=True)
                except: 
                    print("Can not read the pose. ")

            except: 
                print("Excaption happened in reading the dataset or the poses. ") 


              
