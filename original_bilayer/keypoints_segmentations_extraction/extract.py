"""
This file generates a dataset from videos. The main dataset structure that we use is the voxceleb2 dataset structure.

Inputs
----------

Inputs are set in extract.sh file as the following:

    --pretrained_weights_dir <PATH_TO_YOUR_PRETRAINED_WEIGHTS> 
    --video_root <PATH_TO_YOUR_VIDEO_ROOT> 
    --sampling_rate <YOUR_SAMPLING_RATE> 
    --phase <'train' or 'test> 
    --batch_size <YOUR_BATCH_SIZE> 
    --data_root <PATH_TO_WHERE_YOU_WANT_TO_SAVE_DATASET>  
    --output_segmentation True 
    --num_gpus <YOUR_NUM_GPUS>

The video files should be in the format of: VIDEO_ROOT/PERSON_ID/VIDEO_ID/SEQUENCE_ID[.mp4]

Example of video structure:
        
        VIDEO_ROOT _ id00012 _ abc _ 00001.mp4
                   |         |
                   |         |_ def  _ 00001.mp4
                   |                |_ 00002.mp4
                   |               
                   |_ id00013 _ lmn _ 00001.mp4
                   |          |
                   |          |_ opq  _ 00001.mp4
                   |                 |_ 00002.mp4
                   |                 |_ 00003.mp4
                   |
                   |_ id00014 _ rst _ 00001.mp4
                              |    |_ 00002.mp4
                              |
                              |_ uvw  _ 00001.mp4
                                     |_ 00002.mp4
                                     |_ 00003.mp4



Outputs
----------

The output is a dataset in the format of: 
DATA_ROOT/[imgs, keypoints, segs]/[train, test]/PERSON_ID/VIDEO_ID/SEQUENCE_ID/FRAME_NUM[.jpg, .npy, .png]

Example of the dataset structure:

                 DATA_ROOT - [imgs, keypoints, segs] _ phase _ id00012 _ abc _ 00001 _ 0 [.jpg, .npy, .png]
                                                            |         |            |_ 1 [.jpg, .npy, .png]
                                                            |         |            |_ ...
                                                            |         |            |_ 99 [.jpg, .npy, .png]
                                                            |         |
                                                            |         |_ def  _ 00001 _ 0 [.jpg, .npy, .png]
                                                            |                |       |_ 1 [.jpg, .npy, .png]
                                                            |                |       |_ ...
                                                            |                |       |_ 150 [.jpg, .npy, .png]
                                                            |                |
                                                            |                |_ 00002 _ 0 [.jpg, .npy, .png]
                                                            |                        |_ 1 [.jpg, .npy, .png]
                                                            |                        |_ ... 
                                                            |                        |_ 89 [.jpg, .npy, .png]
                                                            |               
                                                            |_ id00013 _ lmn _ 00001 _ 0 [.jpg, .npy, .png]
                                                            |          |             |_ 1 [.jpg, .npy, .png]
                                                            |          |             |_ ... 
                                                            |          |             |_ 89 [.jpg, .npy, .png]
                                                            |          |
                                                            |          |_ opq  _ 00001 _ ...
                                                            |                 |_ 00002 _ ...
                                                            |                 |_ 00003 _ ...
                                                            |
                                                            |_ id00014 _ rst _ 00001 _ ...
                                                                        |    |_ 00002 _ ...
                                                                        |
                                                                        |_ uvw  _ 00001 _ 0 [.jpg, .npy, .png]
                                                                                |       |_ 1 [.jpg, .npy, .png]
                                                                                |       |_ ... 
                                                                                |       |_ 68 [.jpg, .npy, .png]
                                                                                |
                                                                                |_ 00002 _ 0 [.jpg, .npy, .png]
                                                                                |       |_ ...
                                                                                |       |_ 299 [.jpg, .npy, .png]
                                                                                |
                                                                                |_ 00003 _ 0 [.jpg, .npy, .png]
                                                                                        |_ ...
                                                                                        |_ 100 [.jpg, .npy, .png]

"""

# Importing libraries
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

class KeypointSegmentationGenerator():
    @staticmethod
    def get_args(parser):
        # Common properties
        parser.add('--data_root',               default="/video-conf/scratch/pantea/temp_extracts", type=str,
                                                help='root directory to save the dataset')

        parser.add('--phase',                   default='test', type=str,
                                                help='train or test phase')
        
        parser.add('--video_root',              default="/video-conf/vedantha/voxceleb2/dev/mp4/", type=str,
                                                help='root directory of the raw videos')                                                                                                     
        
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

    def __init__(self, args):        

        # Retriving inputs from arguments
        self.args = args
        self.phase = self.args.phase  # could either be test or train

        # Data paths
        self.video_dir = pathlib.Path(self.args.video_root) 
        self.imgs_dir = pathlib.Path(self.args.data_root) / 'imgs' / self.phase
        self.pose_dir = pathlib.Path(self.args.data_root) / 'keypoints' / self.phase

        print("imgs_dir", self.imgs_dir)
        print("pose_dir", self.pose_dir)

        # Video sequences list
        # Finding all of the video ids (or simply all the PERSON_ID/VIDEO_ID combinations in the VIDEO_ROOT/PERSON_ID/VIDEO_ID/)
        sequences = self.video_dir.glob('*/*')
        self.sequences = ['/'.join(str(seq).split('/')[-2:]) for seq in sequences]
        
        # self.sequences is the relative address of all videos in under VIDEO_ROOT
        print(self.sequences)
        
        if args.output_segmentation:
            self.segs_dir = pathlib.Path(self.args.data_root) / 'segs' / self.phase
        
        self.to_tensor = transforms.ToTensor()

        # Stickman/facemasks drawer
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

        # Segmentation wrapper
        if self.args.output_segmentation:
            self.net_seg = wrapper.SegmentationWrapper(self.args)


    def preprocess_data(self, input_imgs, crop_data=True):
        """Generates dataset images, keypoints (also called poses), and segmenatations from input_imgs/ frames

        Inputs
        ----------
        input_imgs: list of images
        crop_data : A flag used center-crop output images and poses (the original paper used crop_data=True,
                    so for consistency we use crop_data=True as well) 

        Returns
        -------
        poses: tensor of keypoints 
        imgs : tensor of images 
        segs : tensor of segmentations

        """        
        imgs = []
        poses = []
        segs = []
        imgs_segs = []

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

            if input_imgs is None:
                if crop_data:
                    # Crop poses
                    s = size * 2
                    pose -= center - size

            else:
                
                img = Image.fromarray(np.array(input_imgs[i]))

                if crop_data:
                    # Crop images and poses
                    img = img.crop((center[0]-size, center[1]-size, center[0]+size, center[1]+size))
                    s = img.size[0]
                    pose -= center - size
                
                # Resizing the image before storing it. If the image is small, this action would add black border around the image
                img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
                imgs.append((self.to_tensor(img)))
                
                # The images that are used to find segmentations, are 
                if self.args.output_segmentation:
                    imgs_segs.append((self.to_tensor(img) - 0.5) * 2)
            
            # This following action (scaling the poses) is done in training pipeline, and should not be done for generating the dataset. 
            if crop_data:
                # This sets the range of pose to 0-256. This is what is needed for voxceleb.py 
                pose = args.image_size*pose / float(s)
            ## poses.append(torch.from_numpy((pose - 0.5) * 2).view(-1))            
            poses.append(torch.from_numpy((pose)).view(-1))

        # Stack the poses from different images
        poses = torch.stack(poses, 0)[None]

        # Use cuda if possible
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
        """Stores dataset images, keypoints (also called poses), and segmenatations from input_imgs/ frames of videos

        Inputs
        ----------
        Arguments in parser

        Returns
        -------
        Prints the percentage of progress in the code
        Prints the video sequene in progress
        Prints the success or the failure of finding the poses, images, or segmentations

        """ 
        # For loop over all the videos in the VIDEO_ROOT
        for index in range(0,len(self.sequences)):
            print("Progress Percentage: ",str(index/len(self.sequences)*100))
            print("Sequences is: ", self.sequences[index])

            # Get all the video sequences in VIDEO_ROOT/PERSON_ID/VIDEO_ID [index]
            # self.sequences[index] = VIDEO_ROOT/PERSON_ID/VIDEO_ID [index]
            filenames_vid = list((self.video_dir / self.sequences[index]).glob('*'))
            filenames_vid = [pathlib.Path(*filename.parts[-3:]).with_suffix('') for filename in filenames_vid]

            # filenames are the name of all the video files in VIDEO_ROOT/PERSON_ID/VIDEO_ID [index] directory
            filenames = list(set(filenames_vid))
            filenames = sorted(filenames)

            # Iterate over all the videos in the folder
            for filename in filenames:

                # The path to the video: VIDEO_ROOT/PERSON_ID/VIDEO_ID [index]/ filename
                video_path = pathlib.Path(self.video_dir) / filename.with_suffix('.mp4')
                video = cv2.VideoCapture(str(video_path))
                frame_num = 0

                # Capturing the video frames 
                while video.isOpened():
                    ret, frame = video.read()
                    print(frame_num)
                    if frame is None:
                        break

                    # Sample the frames with sampling_rate argument
                    if frame_num % self.args.sampling_rate != 0:
                        pass
                    
                    else:
                        # Reformat to proper RGB/BGR
                        frame = frame[:,:,::-1]
                        
                        # Find the keypoints and segmentations of the frame, process frame by frame 
                        try: 
                            poses, imgs, segs = self.preprocess_data(frame, crop_data=True)
                            if poses is not None and len(poses) == 1:
                                # Paths to save imgs and segs
                                imgs_path = str(self.imgs_dir) +"/"+ str(filename) + "/" + str(frame_num)
                                keypoints_path = str(self.pose_dir) +"/"+ str(filename) + "/" + str(frame_num)
                                # Make the corresponding directories to save dataset
                                os.makedirs(str(self.imgs_dir) +"/"+ str(filename), exist_ok=True)
                                os.makedirs(str(self.pose_dir) +"/"+ str(filename), exist_ok=True)
                                # Save the images and keypoints
                                save_image(imgs[0,0,:,:,:], imgs_path + '.jpg')
                                np.save(keypoints_path , poses[0,:,:].cpu().numpy())
                                if self.args.output_segmentation:
                                    # Saving the segmentations
                                    segs_path = str(self.segs_dir) +"/"+ str(filename) + "/" + str(frame_num)
                                    os.makedirs(str(self.segs_dir) +"/"+ str(filename), exist_ok=True)
                                    save_image(segs[0,0,:,:,:], segs_path + '.png')

                        except Exception as e:
                            raise(e) 
                            print("Excaption happened in reading the poses of the frame.")

                    frame_num+=1
                video.release()
                print("Saved images of ", str(video_path), " in ", str(self.imgs_dir) ,"/", str(filename))
              
if __name__ == "__main__":

    # Parse options 
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add = parser.add_argument

    # Instanciate the keypoint-segmentation generator
    KeypointSegmentationGenerator.get_args(parser)

    args, _ = parser.parse_known_args()

    # initialize the model and save the imgs, poses, and segmentations
    generator = KeypointSegmentationGenerator(args)
    generator.get_poses()
