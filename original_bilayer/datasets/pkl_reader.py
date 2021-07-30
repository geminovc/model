"""
This file is for picking the source and target frames in the same session that:
if dataset_method = 'l2_distance': their keypoints have close l2 distance less than a threshold (close_keypoints_threshold),
if dataset_method = 'difficult_pose': are considered difficult poses.


Before using this dataloader, you need to need to preprocess the dataset using the find_l2_distance or find_difficult_poses module (based on your choice of dataset_method).
The preprocess modules are in ../pose_analysis. You need to enter the <results_folder/dataset_name> in those modules as an input here in root_to_pkl_keypoints. 
The structure of input dataset, which is stored in root_to_pkl_keypoints, is:
ROOT_TO_PKL_KEYPOINTS/[train, test]/PERSON_ID/VIDEO_ID/SEQUENCE_ID/<L2_Distances or Difficult_poses>.pkl
Each L2_Distances.pkl file contains a dictionary with the following structure:

{
('-1','-1'): 'data_root of the dataset that the keypoints belong to',
('frame_num_1','frame_num_2'): 'L2_distance (keypoint_1, keypoint_2)',
...
}

Each Difficult_poses.pkl file contains a dictionary with the following structure:
 
{
('-1'): 'data_root of the dataset that the keypoints belong to',
('0'): 'frame_num',
..

}


Note that the last folder of ROOT_TO_PKL_KEYPOINTS has the same name as DATA_ROOT, meaning that the L2 distances of ROOT_TO_PKL_KEYPOINTS belong to datset in DATA_ROOT. 

Example:
ROOT_TO_PKL_KEYPOINTS = /data/pantea/close_keypoints/per_person_extracts
DATA_ROOT = /video-conf/scratch/pantea/per_person_extracts

In pose_analysis module in ../pose_analysis we have:
ROOT_TO_PKL_KEYPOINTS = RESULTS_FOLDER/DATASET_NAME

Arguments
----------

root_to_pkl_keypoints: The root directory to where the L2_distances of the input data_root is stored. 
        
close_keypoints_threshold: The threshold with which we call two keypoints close. 

dataset_method: l2_distance or difficult_pose

Outputs
----------

The output is data_dict which contains source and target images with L2 keypoint distance of less than close_keypoints_threshold,
or with difficult poses depending on dataset_method.
The script is faithful to the papers implementation and it picks one random source/target pair:
if dataset_method = 'l2_distance': with keypoints having L2 distance of less than close_keypoints_threshold, from each video.
if dataset_method = 'difficult_poses': which both are considered images with difficult poses. 

"""

import torch
from torch.utils import data
from torchvision import transforms
import glob
import pathlib
from PIL import Image
import numpy as np
import pickle as pkl
import cv2
import random
import math
import pdb
from datasets import utils as ds_utils
from runners import utils as rn_utils
import pickle


class DatasetWrapper(data.Dataset):
    @staticmethod
    def get_args(parser):
        # Common properties
        parser.add('--experiment_dir',           default='.', type=str,
                                                 help='directory to save logs')

        parser.add('--experiment_name',          default='test', type=str,
                                                 help='name of the experiment used for logging')

        parser.add('--num_source_frames',        default=1, type=int,
                                                 help='number of frames used for initialization of the model')

        parser.add('--num_target_frames',        default=1, type=int,
                                                 help='number of frames per identity used for training')

        parser.add('--image_size',               default=256, type=int,
                                                 help='output image size in the model')

        parser.add('--num_keypoints',            default=68, type=int,
                                                 help='number of keypoints (depends on keypoints detector)')

        parser.add('--output_segmentation',      default='True', type=rn_utils.str2bool, choices=[True, False],
                                                 help='read segmentation mask')

        parser.add('--output_stickmen',          default='True', type=rn_utils.str2bool, choices=[True, False],
                                                 help='draw stickmen using keypoints')
        
        parser.add('--stickmen_thickness',       default=2, type=int, 
                                                 help='thickness of lines in the stickman')

        # Paired target and source 

        parser.add('--dataset_method',           default='l2_distance', type=str,
                                                 help='could be l2_distance or difficult_pose')
                                     
        parser.add('--root_to_pkl_keypoints',    default='/video-conf/scratch/pantea/pose_results/close_l2_keypoints/temp_per_person_extracts', type=str, 
                                                 help='If True, the images, keypoints, and segs are picked from files')
        
        parser.add('--close_keypoints_threshold',default=500, type=float,
                                                 help='threshold of defining two close keypoints')

        parser.add('--mask_source_and_target',   default='True', type=rn_utils.str2bool, choices=[True, False],
                                                 help='mask the souce and target from the beginning')
        
        parser.add('--same_source_and_target',   default='False', type=rn_utils.str2bool, choices=[True, False],
                                                 help='set source = target in the experminet')         
        return parser

    def __init__(self, args, phase, pose_component = 'none'):
        super(DatasetWrapper, self).__init__()
        # Store options
        self.phase = phase
        self.args = args

        self.to_tensor = transforms.ToTensor()
        self.epoch = 0 if args.which_epoch == 'none' else int(args.which_epoch)
        data_root = args.data_root

        if phase == 'metrics':
            data_root = args.metrics_root
        
        self.imgs_dir = pathlib.Path(data_root) / 'imgs' / phase
        self.pose_dir = pathlib.Path(data_root) / 'keypoints' / phase

        if args.output_segmentation:
            self.segs_dir = pathlib.Path(data_root) / 'segs' / phase
        
        self.pkl_keypoints_dir = self.args.root_to_pkl_keypoints + '/' + self.phase
        
        # Video sequences list
        # Please don't mix the sequence, which is the total number of videos, with sessions (which are essentially short video clips).
        # Each sequence, is made up of multiple sessions.
        sequences = pathlib.Path(self.pkl_keypoints_dir).glob('*/*')
        self.sequences = ['/'.join(str(seq).split('/')[-2:]) for seq in sequences]

        # If you are using metrics, sort the Sequences so it's easier
        # to keep track of them between runs
        if phase == 'metrics':
            self.sequences = sorted(self.sequences)
        
        print(len(self.sequences), self.sequences)

        # Prepare experiment directories and save options
        experiment_dir = pathlib.Path(args.experiment_dir)
        self.experiment_dir = experiment_dir / 'runs' / args.experiment_name

    # Load the pickle files as dictionary
    def load_pickle(self, path_string):
        pkl_file = open(path_string, 'rb')
        my_dict = pickle.load(pkl_file)
        pkl_file.close()
        return my_dict

    # find pairs of frames in the session whose keypoint l2 distance is lower than a threshold
    def find_frame_pairs_with_close_keypoints (self, keypoints_dict, threshold):
        # The ('-1','-1') key contains the data_root of the keypoints
        keypoints_dict.pop(('-1', '-1'), None)
        return [k for k,v in keypoints_dict.items() if float(v) >= threshold and k!= ('-1', '-1')]
    
    # find the frame numbers in the session with difficult pose
    def find_frames_with_difficult_poses (self, keypoints_dict):
        # The ('-1') key contains the data_root of the keypoints
        keypoints_dict.pop(('-1'), None)
        return [v for k,v in keypoints_dict.items()]

    def __getitem__(self, index):
        
        # No pkl keypoints for metrics loader
        if self.phase == 'metrics':
            while True:
                count+=1
                try:
                    filenames_img = list((self.imgs_dir / self.sequences[index]).glob('*/*'))
                    filenames_img = [pathlib.Path(*filename.parts[-4:]).with_suffix('') for filename in filenames_img]

                    filenames_npy = list((self.pose_dir / self.sequences[index]).glob('*/*'))
                    filenames_npy = [pathlib.Path(*filename.parts[-4:]).with_suffix('') for filename in filenames_npy]

                    filenames = list(set(filenames_img).intersection(set(filenames_npy)))

                    if self.args.output_segmentation:
                        filenames_seg = list((self.segs_dir / self.sequences[index]).glob('*/*'))
                        filenames_seg = [pathlib.Path(*filename.parts[-4:]).with_suffix('') for filename in filenames_seg]

                        filenames = list(set(filenames).intersection(set(filenames_seg)))
                    
                    if len(filenames)!=0:
                        break
                    else:
                        raise # the length of filenames is zero.

                except Exception as e:
                    print("# Exception is raised if filenames list is empty or there was an error during read")
                    index = (index + 1) % len(self)

            filenames = sorted(filenames)
        
        # If 'train' or 'test' phase, find pairs with close keypoints
        else: 
            # Sample source and target frames for the current video sequence
            filenames = []

            # Load all pickle file for the current video 
            keypoints_pickles = list(pathlib.Path(self.pkl_keypoints_dir + '/' + self.sequences[index]).glob('*/*'))
            
            # Sample one session from the video
            keypoints_pickle_path = random.choice(keypoints_pickles)
            keypoints_dict =  self.load_pickle(keypoints_pickle_path)

            if self.args.dataset_method == 'l2_distance':
                close_keys = self.find_frame_pairs_with_close_keypoints(keypoints_dict, self.args.close_keypoints_threshold)
                # The source and target relative paths (sample one pair from the close keypoints in a session)
                relative_path = '/'.join(str(keypoints_pickle_path).split('/')[-4:-1])
                source_target_pair = random.choice(close_keys)
            
            elif self.args.dataset_method == 'difficult_pose':
                difficult_frames = self.find_frames_with_difficult_poses(keypoints_dict)
                # The source and target relative paths (sample one pair from the close keypoints in a session)
                relative_path = '/'.join(str(difficult_pose_pickle_path).split('/')[-4:-1])
                source_target_pair = random.sample(difficult_frames, 2)

            filenames = [pathlib.Path(relative_path + '/' + source_target_pair[0]), pathlib.Path(relative_path + '/' + source_target_pair[1])]
            if self.args.same_source_and_target:
                filenames = [pathlib.Path(relative_path + '/' + source_target_pair[0]), pathlib.Path(relative_path + '/' + source_target_pair[0])]
            
            random.shuffle(filenames)
            print(filenames)



        imgs = []
        poses = []
        stickmen = []
        segs = []

        reserve_index = -1 # take this element of the sequence if loading fails
        sample_from_reserve = False

        # Read and pre-process imgs, keypoints, and segs if available        
        while len(imgs) < self.args.num_source_frames + self.args.num_target_frames:
            if reserve_index == len(filenames):
                raise # each element of the filenames list is unavailable for load

            # Sample a frame number
            if sample_from_reserve:
                filename = filenames[reserve_index]

            # Added shuffle to the filenames for test and train phase
            else:
                frame_num = len(imgs)
                filename = filenames[frame_num]

            # Read images
            img_path = pathlib.Path(self.imgs_dir) / filename.with_suffix('.jpg')
            try:
                img = Image.open(img_path)
                # Preprocess an image
                s = img.size[0]
                img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
            except:
                sample_from_reserve = True
                reserve_index += 1
                continue

            imgs += [self.to_tensor(img)]

            # Read keypoints
            keypoints_path = pathlib.Path(self.pose_dir) / filename.with_suffix('.npy')
            try:
                keypoints = np.load(keypoints_path).astype('float32')
            except:
                imgs.pop(-1)
                sample_from_reserve = True
                reserve_index += 1
                continue

            # Normalize the keypoints to (0,1) range 
            keypoints = keypoints.reshape((68,2)) 
            keypoints = keypoints[:self.args.num_keypoints, :]
            keypoints[:, :2] /= s
            keypoints = keypoints[:, :2]
            
            # Reshape the keypoints to feed the network
            poses += [torch.from_numpy(keypoints.reshape(-1))]

            if self.args.output_segmentation:
                seg_path = pathlib.Path(self.segs_dir) / filename.with_suffix('.png')

                try:
                    seg = Image.open(seg_path)
                    seg = seg.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
                except:
                    imgs.pop(-1)
                    poses.pop(-1)
                    sample_from_reserve = True
                    reserve_index += 1
                    continue
                # Convert 3-channel segmentations to 1 grayscale image used to be segs += [self.to_tensor(seg)]
                segs += [self.to_tensor(seg)[0][None]]

            sample_from_reserve = False
        
        # Normalized the images and poses in (-1,1) range
        imgs = (torch.stack(imgs)- 0.5) * 2.0
        poses = (torch.stack(poses) - 0.5) * 2.0

        if self.args.output_stickmen:
            stickmen = ds_utils.draw_stickmen(self.args, poses)

        if self.args.output_segmentation:
            segs = torch.stack(segs)

        # Assigning the source and target images in the data_dict with the correct key
        data_dict = {}
        if self.args.num_source_frames:
            data_dict['source_imgs'] = imgs[:self.args.num_source_frames]
        data_dict['target_imgs'] = imgs[self.args.num_source_frames:]
        
        if self.args.num_source_frames:
            data_dict['source_poses'] = poses[:self.args.num_source_frames]
        data_dict['target_poses'] = poses[self.args.num_source_frames:]

        if self.args.output_stickmen:
            if self.args.num_source_frames:
                data_dict['source_stickmen'] = stickmen[:self.args.num_source_frames]
            data_dict['target_stickmen'] = stickmen[self.args.num_source_frames:]
        
        if self.args.output_segmentation:
            if self.args.num_source_frames:
                data_dict['source_segs'] = segs[:self.args.num_source_frames]
            data_dict['target_segs'] = segs[self.args.num_source_frames:]

        if self.args.mask_source_and_target and self.args.output_segmentation:
            data_dict['source_imgs'] = data_dict['source_imgs'] * data_dict['source_segs'] + (-1) * (1 - data_dict['source_segs'])
            data_dict['target_imgs'] = data_dict['target_imgs'] * data_dict['target_segs'] + (-1) * (1 - data_dict['target_segs'])     

        data_dict['indices'] = torch.LongTensor([index])

        return data_dict

    def __len__(self):
        return len(self.sequences)

    # To shuffle the dataset before the epoch
    def shuffle(self):
        if self.phase != 'metrics': # Don't shuffle metrics
            self.sequences = [self.sequences[i] for i in torch.randperm(len(self.sequences)).tolist()]
