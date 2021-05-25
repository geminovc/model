
import argparse
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import sys
sys.path.append("../")
import pathlib
import numpy as np
import cv2
import importlib
import ssl
from datasets import utils as ds_utils
from runners import utils as rn_utils
from external.Graphonomy import wrapper
import face_alignment
from torchvision.utils import save_image

class keypoint_segmentation_generator():
    @staticmethod
    def get_args(parser):
        # Common properties
        parser.add('--data_root',               default="/video-conf/scratch/pantea/temp_extracts", type=str,
                                                help='root directory of the train data')
        
        parser.add('--keypoint_dir',            default="keypoint", type=str,
                                                help='root directory of the stored keypoints')
        
        parser.add('--video_root',              default="/video-conf/vedantha/voxceleb2/dev/mp4/", type=str,
                                                help='root directory of the raw videos')        
        
        parser.add('--segmentatio_dir',         default="segs", type=str,
                                                help='root directory of the data')                                                                                             
        
        parser.add('--num_source_frames',       default=1, type=int,
                                                help='number of frames used for initialization of the model')
        
        parser.add('--pretrained_weights_dir',  default='/video_conf/scratch/pantea', type=str,
                                                help='directory for pretrained weights of loss networks (lpips , ...)')
 
        parser.add('--num_target_frames',       default=1, type=int,
                                                help='number of frames per identity used for training')

        parser.add('--image_size',              default=256, type=int,
                                                help='output image size in the model')

        parser.add('--num_keypoints',           default=68, type=int,
                                                help='number of keypoints (depends on keypoints detector)')

        parser.add('--output_segmentation',     default='True', type=rn_utils.str2bool, choices=[True, False],
                                                help='read segmentation mask')

        parser.add('--output_stickmen',         default='False', type=rn_utils.str2bool, choices=[True, False],
                                                help='draw stickmen using keypoints')
        
        parser.add('--stickmen_thickness',      default=2, type=int, 
                                                help='thickness of lines in the stickman')

        parser.add('--num_gpus',                default=1, type=int, 
                                                help='thickness of lines in the stickman')

        parser.add('--sampling_rate',           default=1, type=int, 
                                                help='sampling rate for extracting the frames from videos')

        return parser

    def __init__(self, args, phase):        
        # Store options
        self.phase = phase
        self.args = args


        self.to_tensor = transforms.ToTensor()

        data_root = self.args.data_root
        
        # Data paths
        self.video_dir = pathlib.Path("/video-conf/scratch/pantea/temp_dataset/") #'/video-conf/scratch/pantea/0
        self.imgs_dir = pathlib.Path(data_root) / 'imgs' / phase
        self.pose_dir = pathlib.Path(data_root) / 'keypoints' / phase

        print("imgs_dir", self.imgs_dir)
        print("pose_dir", self.pose_dir)

        # Video sequences list
        sequences = self.video_dir.glob('*/*')
        self.sequences = ['/'.join(str(seq).split('/')[-2:]) for seq in sequences]
        
        if args.output_segmentation:
            self.segs_dir = pathlib.Path(data_root) / 'segs' / phase
        
        self.to_tensor = transforms.ToTensor()

        # Stickman/facemasks drawer
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

        # Segmentation wrapper
        if self.args.output_segmentation:
            self.net_seg = wrapper.SegmentationWrapper(self.args)


    def preprocess_data(self, input_imgs, crop_data=True):
        imgs = []
        poses = []
        segs = []
        imgs_segs = []

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
                
                img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
                imgs.append((self.to_tensor(img)))
                if self.args.output_segmentation:
                    imgs_segs.append((self.to_tensor(img) - 0.5) * 2)
            
            # Must not remove comment, done in the voxceleb2.py
            # if crop_data:
            #     pose = pose / float(s)
            
            poses.append(torch.from_numpy((pose)).view(-1))
            # Must not remove comment, normalization in the voxceleb2.py            
            # poses.append(torch.from_numpy((pose - 0.5) * 2).view(-1))

        poses = torch.stack(poses, 0)[None]


        if input_imgs is not None:
            imgs = torch.stack(imgs, 0)[None]

        if self.args.num_gpus > 0:
            poses = poses.cuda()
            
            if input_imgs is not None:
                imgs = imgs.cuda()

        
        if input_imgs is not None and self.args.output_segmentation:
            imgs_segs = torch.stack(imgs_segs, 0)[None]
            if self.args.num_gpus > 0:        
                imgs_segs = imgs_segs.cuda()

        # Get the segmentations
        segs = None
        if hasattr(self, 'net_seg') and not isinstance(imgs, list):
            segs = self.net_seg(imgs_segs)[None]
        

        return poses, imgs, segs

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
                    if frame_num % self.args.sampling_rate != 0:
                        print("skipping frame number: ",frame_num)
                    
                    else:
                        frame = frame[:,:,::-1]
                        try: 
                            poses, imgs, segs = self.preprocess_data(frame, crop_data=True)
                            if poses is not None and len(poses) == 1:
                                imgs_path = str(self.imgs_dir) +"/"+ str(filename) + "/" + str(frame_num)
                                keypoints_path = str(self.pose_dir) +"/"+ str(filename) + "/" + str(frame_num)
                                os.makedirs(str(self.imgs_dir) +"/"+ str(filename), exist_ok=True)
                                os.makedirs(str(self.pose_dir) +"/"+ str(filename), exist_ok=True)
                                temp = imgs[0,0,:,:,:]
                                save_image(temp, imgs_path + '.jpg')
                                np.save(keypoints_path , poses[0,:,:].cpu().numpy())
                                if self.args.output_segmentation:
                                    segs_path = str(self.segs_dir) +"/"+ str(filename) + "/" + str(frame_num)
                                    os.makedirs(str(self.segs_dir) +"/"+ str(filename), exist_ok=True)
                                    save_image(segs[0,0,:,:,:], segs_path + '.png')

                        except: 
                            print("Excaption happened in reading the poses of the frame.") 

                    frame_num+=1
                video.release()
                print("Extraction finished for ", str(video_path))
              
if __name__ == "__main__":
    ## Parse options ##
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add = parser.add_argument

    keypoint_segmentation_generator.get_args(parser)

    args, _ = parser.parse_known_args()

    # initialize the model
    generator = keypoint_segmentation_generator(args, 'train')
    generator.get_poses()