
import argparse
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import sys
sys.path.append('../')
import pathlib
import numpy as np
import cv2
import importlib
import ssl
import time
from datasets import utils as ds_utils
from runners import utils as rn_utils
import face_alignment
from torch.utils.data import DataLoader
from external.Graphonomy import wrapper
from keypoints_dataset import Dataset


class keypoint_segmentation_generator(nn.Module):
    @staticmethod
    def get_args(parser):
        # Common properties
        parser.add('--data_root',               default="/data/pantea/video_conf", type=str,
                                                help='root directory of the train data')
 
        parser.add('--phase',                   default='test', type=str,
                                                help='test or train dataset')

        parser.add('--pretrained_weights_dir',  default='/video_conf/scratch/pantea', type=str,
                                                help='directory for pretrained weights of loss networks (lpips , ...)')
 
        parser.add('--keypoint_dir',            default="keypoints", type=str,
                                                help='root directory of the stored keypoints')

        parser.add('--video_root',              default="/video-conf/vedantha/voxceleb2/dev/mp4/", type=str,
                                                help='root directory of the raw videos')        

        parser.add('--segmentatio_dir',         default="segs", type=str,
                                                help='root directory of the data')                                                                                             

        parser.add('--num_source_frames',       default=1, type=int,
                                                help='number of frames used for initialization of the model')

        parser.add('--num_target_frames',       default=1, type=int,
                                                help='number of frames per identity used for training')

        parser.add('--image_size',              default=256, type=int,
                                                help='output image size in the model')

        parser.add('--batch_size',              default=1, type=int,
                                                help='batch size across all GPUs')

        parser.add('--num_keypoints',           default=68, type=int,
                                                help='number of keypoints (depends on keypoints detector)')

        parser.add('--output_segmentation',     default='True', type=rn_utils.str2bool, choices=[True, False],
                                                help='read segmentation mask')

        parser.add('--output_stickmen',         default='False', type=rn_utils.str2bool, choices=[True, False],
                                                help='draw stickmen using keypoints')
        
        parser.add('--stickmen_thickness',      default=2, type=int, 
                                                help='thickness of lines in the stickman')
        
        parser.add('--num_gpus',                default=1, type=int, 
                                                help='number of gpus that we use')
        
        parser.add('--sampling_rate',           default=1, type=int, 
                                                help='sampling rate for extracting the frames from videos')

        # Dataset options
        args, _ = parser.parse_known_args()

        return parser

    def __init__(self, args):
        super(keypoint_segmentation_generator, self).__init__()        
        # Store options
        self.args = args
        self.phase = str(self.args.phase)
        self.frame_count = 0


        self.to_tensor = transforms.ToTensor()

        data_root = self.args.data_root
        video_dir = self.args.video_root

        # Data paths
        self.video_dir = pathlib.Path(video_dir) 
        self.imgs_dir = pathlib.Path(data_root) / 'imgs' / self.phase
        self.pose_dir = pathlib.Path(data_root) / 'keypoints' / self.phase

        print("imgs_dir", self.imgs_dir)
        print("pose_dir", self.pose_dir)

        # Video sequences list
        print("Getting the sequences ... This might take some time.")
        sequences = self.video_dir.glob('*/*')
        self.sequences = ['/'.join(str(seq).split('/')[-2:]) for seq in sequences]
        
        print("Retrieved the sequences.")
        if args.output_segmentation:
            self.segs_dir = pathlib.Path(data_root) / 'segs' / self.phase
            self.net_seg = wrapper.SegmentationWrapper(self.args)
        self.to_tensor = transforms.ToTensor()

        # Stickman/facemasks drawer
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

        if self.args.num_gpus>0:
            self.cuda()



    def preprocess_data(self, input_imgs, crop_data=True):
        imgs = []
        poses = []
        segs = []
        stickmen = []
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
                
                if self.args.output_segmentation:
                    ## here I decided to use the non-croped version of images for segmentations
                    ## The croped version of images would give a segmentation that has a black stripe on top
                    img_segs = Image.fromarray(np.array(input_imgs[i]))
                    img_segs = img_segs.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
                    imgs_segs.append((self.to_tensor(img_segs) - 0.5) * 2)


                
                # Crop images and poses
                img = Image.fromarray(np.array(input_imgs[i]))

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

        
        if input_imgs is not None and self.args.output_segmentation:
            imgs_segs = torch.stack(imgs_segs, 0)[None]
            if self.args.num_gpus > 0:        
                imgs_segs = imgs_segs.cuda()

        # get the segmentations
        segs = None
        if hasattr(self, 'net_seg') and not isinstance(imgs, list):
            segs = self.net_seg(imgs_segs)[None]
        

        return poses, imgs, segs, stickmen

    def get_poses (self):
        dataset = Dataset(self.args.sampling_rate, self.video_dir, self.sequences)
        dataloader = DataLoader(dataset, self.args.batch_size, shuffle=False)
        self.seq_id = 0
        while int(self.seq_id) < len(self.sequences):
            try:
                self.start = time.time()
                output = next(iter(dataloader))
                frames , frame_nums, filenames, seq_ids = output[0],output[1],output[2], output[3]
                self.seq_id = seq_ids.max()
                #self.frame_count+=1
                print("Reading rate is (frame/sec):", self.args.batch_size/(time.time()-self.start))
                try: 
                    poses, imgs, segs_tensors, stickmen = self.preprocess_data(frames, crop_data=True)
                    print("got poses, frames, and segs for this batch")
                    poses = np.array(poses.cpu())
                    frames = (frames.cpu()).numpy()
                    number_of_poses = poses.shape[1]
                    for i in range(0, number_of_poses):
                        pose = poses[:,i,:]
                        frame = frames[i,:,:,:]
                        segs_t = segs_tensors [:,i,:,:,:] 
                        if pose is not None:
                            imgs_save_path = str(self.imgs_dir) +"/"+ str(filenames[i]) + "/" + str(int(frame_nums[i]))
                            keypoints_save_path = str(self.pose_dir) +"/"+ str(filenames[i]) + "/" + str(int(frame_nums[i]))
                            os.makedirs(str(self.imgs_dir) +"/"+ str(filenames[i]), exist_ok=True)
                            os.makedirs(str(self.pose_dir) +"/"+ str(filenames[i]), exist_ok=True)
                            frame = frame[:,:,::-1]
                            img = Image.fromarray(frame)
                            img.save(imgs_save_path + '.jpg')
                            np.save(keypoints_save_path, pose)
                            print("saved images to:", imgs_save_path)
                            if self.args.output_segmentation:
                                segs_save_path = str(self.segs_dir) +"/"+ str(filenames[i]) + "/" + str(int(frame_nums[i]))
                                os.makedirs(str(self.segs_dir) +"/"+ str(filenames[i]), exist_ok=True)
                                data = ((segs_t.cpu()).numpy()).reshape(256,256)
                                rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
                                segs = Image.fromarray(rescaled)
                                segs.save(segs_save_path + '.png')
                                print("saved segmentation to:", segs_save_path)
                except: 
                        print("Can not read the poses or segmentations.")

            except: 
                print("Excaption happened in reading the dataset or the poses. ") 


              
if __name__ == "__main__":
    ## Parse options ##
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add = parser.add_argument

    keypoint_segmentation_generator.get_args(parser)

    args, _ = parser.parse_known_args()

    # initialize the model
    generator = keypoint_segmentation_generator(args)
    generator.get_poses()