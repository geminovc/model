"""
This file is for picking the source and target frames in session that have close yaw. 
Before using this dataloader, you need to do a preprocess on the dataset to find the yaws using the angle_extraction module. 
The preprocess modules are in ../angle_extraction. You need to enter the <results_folder/angles> of those modules as an input here in root_to_yaws. 
The structure of input dataset, which is stored in root_to_yaws, is:
root_to_yaws/phase/PERSON_ID/VIDEO_ID/SEQUENCE_ID/FRAME_NUM.npy 
This numpy file contains array([pitch, yaw, roll]) for the image in DATA_ROOT/phase/PERSON_ID/VIDEO_ID/SEQUENCE_ID/FRAME_NUM.jpg

Arguments
----------

root_to_yaws: The root directory to where the yaws of the input data_root is stored. 
        
yaw_method: The method by which the soucre-target pair is selected based on yaw for training.
    Possible choices: min_max, close_uniform, close_original 
    min_max: selecting source-target pair randomly in each session with abs_min_yaw < yaw < abs_max_yaw 
    close_original: selecting source-target pair from a random session belonging to a random bin in that session 
    close_unifrom: selecting a random bin, then selecting source-target pair from a random session having that bin

abs_min_yaw: The minimum yaw value for min_max method
abs_max_yaw: The maximum yaw value for min_max method

Outputs
----------

The output is data_dict which contains source and target images chosen based on their yaws and the yaw_method. 
The script is faithful to the papers implementation and it picks one random source/target pair: 

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
        # Yaws                              
        parser.add('--root_to_yaws',             default='/video-conf/scratch/pantea/pose_results/yaws/per_person_1_three_datasets/angles', type=str, 
                                                 help='The directory where the yaws are stored in voxceleb2 format')
        
        parser.add('--abs_min_yaw',              default=50, type=float, 
                                                 help='The minimum abs value for yaw')
        
        parser.add('--abs_max_yaw',              default=90, type=float, 
                                                 help='The maximum abs value for yaw') 
        
        parser.add('--mask_source_and_target',   default='True', type=rn_utils.str2bool, choices=[True, False],
                                                 help='Mask the souce and target from the beginning')
        
        parser.add('--yaw_method',               default='close_original', type=str, 
                                                 help='The method by which the soucre-target pair is selected based on yaw for training. \
                                                 Possible choices: min_max, close_uniform, close_original \
                                                 min_max: selecting source-target pair randomly in each session with abs_min_yaw < yaw < abs_max_yaw \
                                                 close_original: selecting source-target pair from a random session belonging to a random bin in that session \
                                                 close_unifrom: selecting a random bin, then selecting source-target pair from a random session having that bin')                                                      
        
        parser.add('--same_source_and_target',   default='False', type=rn_utils.str2bool, choices=[True, False],
                                                 help='mask the source and target from the beginning')
                                                              
        return parser

    def __init__(self, args, phase, pose_component):
        super(DatasetWrapper, self).__init__()
        # Store options
        self.phase = phase
        self.args = args
        self.pose_component = pose_component

        # setting values for pose_realted test/unseen_test dataloaders
        if self.pose_component == 'easy_pose' and self.phase != 'train':
            self.yaw_method = 'min_max'
            self.abs_max_yaw = 10
            self.abs_min_yaw = 0

        elif self.pose_component == 'hard_pose' and self.phase != 'train':
            self.yaw_method = 'min_max'
            self.abs_max_yaw = 90
            self.abs_min_yaw = 50
        
        elif self.pose_component == 'combined_pose' and self.phase != 'train':
            self.yaw_method = 'close_original'
        # For train, select the values from the arguments
        else:
            self.yaw_method = args.yaw_method
            self.abs_max_yaw = args.abs_max_yaw
            self.abs_min_yaw = args.abs_min_yaw
            
        self.to_tensor = transforms.ToTensor()
        self.epoch = 0 if args.which_epoch == 'none' else int(args.which_epoch)

        data_root = args.data_root

        if phase == 'metrics':
            data_root = args.metrics_root
        
        self.imgs_dir = pathlib.Path(data_root) / 'imgs' / phase
        self.pose_dir = pathlib.Path(data_root) / 'keypoints' / phase

        if args.output_segmentation:
            self.segs_dir = pathlib.Path(data_root) / 'segs' / phase
        
        self.yaw_dir = self.args.root_to_yaws + '/' + self.phase
        
        # Video sequences list
        # Please don't mix the sequence, which is the total number of videos, with sessions (which are essentially short video clips).
        # Each sequence, is made up of multiple sessions.

        sequences = pathlib.Path(self.yaw_dir).glob('*/*')
        self.sequences = ['/'.join(str(seq).split('/')[-2:]) for seq in sequences]

        # Since the general dataset is too big, we sample 22 items if sample is chosen
        if self.args.sample_general_dataset and self.args.data_root == self.args.general_data_root:
            self.sequences = random.sample(self.sequences, 22)

        # If you are using metrics, sort the Sequences so it's easier
        # to keep track of them between runs
        if phase == 'metrics':
            self.sequences = sorted(self.sequences)
        
        # Prepare experiment directories and save options
        experiment_dir = pathlib.Path(args.experiment_dir)
        self.experiment_dir = experiment_dir / 'runs' / args.experiment_name

        # Preprocess for each yaw_method
        if self.yaw_method == 'min_max':
            self.min_max_preprocess ()
        elif self.yaw_method == 'close_original':
            self.bins = [[-90,-80], [-80,-70],[-70,-60],[-60,-50],[-50,-40],[-40,-30],[-30,-20],
            [-20,-10],[-10,0],[0,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80],[80,90]]
            self.close_original_preprocess () 
        elif self.yaw_method == 'close_uniform':
            self.bins = [[-90,-80], [-80,-70],[-70,-60],[-60,-50],[-50,-40],[-40,-30],[-30,-20],
            [-20,-10],[-10,0],[0,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80],[80,90]]
            self.close_uniform_preprocess ()
            # allow the dataset, to select the random bin frist, then select a session
            self.change_bin = True
            self.change_current_bin()
        
        print("The dataloader charactristics:", self.phase, self.pose_component, self.yaw_method, self.yaw_dir, len(self.sequences))

    # Preprocess the yaws when yaw_method == 'min_max'
    # This function goes over all the sequences to delete the ones that don't contain yaws in the desired range of (self.abs_min_yaw, self.abs_max_yaw)
    def min_max_preprocess (self):
        temp_sequences = []
        # sequence_session_frames_dict [(sequence , session)] contains the frames in path sequence/session with yaw in range of (self.abs_min_yaw, self.abs_max_yaw)  
        self.sequence_session_frames_dict = {}
        for sequence in self.sequences:
            count = 0
            sessions = sorted(list(pathlib.Path(self.yaw_dir + '/' + sequence).glob('*')))
            for session in sessions:
                yaw_npy_paths = sorted(list(pathlib.Path(str(session)).glob('*')))
                yaw_dict = self.load_session_yaws(yaw_npy_paths)
                difficult_frames_in_session = self.find_min_max_poses(yaw_dict)
                if len(difficult_frames_in_session) >= self.args.num_source_frames + self.args.num_target_frames:
                    self.sequence_session_frames_dict [(sequence , str(session).split('/')[-1])] = difficult_frames_in_session
                    if sequence not in temp_sequences:
                        temp_sequences.append(sequence)
                        count +=1
            if count == 0:
                print(sequence, "does not have yaws in  range (abs_min_yaw, abs_max_yaw).")
        self.sequences = temp_sequences

    # Preprocess the yaws when yaw_method == 'close_origial'
    # This function goes over all the sequences to find existing bins in each session
    def close_original_preprocess (self):
        self.sequence_session_bins_frames_dict = {}
        for sequence in self.sequences:
            sessions = sorted(list(pathlib.Path(self.yaw_dir + '/' + sequence).glob('*')))
            for session in sessions:
                yaw_npy_paths = sorted(list(pathlib.Path(str(session)).glob('*')))
                yaw_dict = self.load_session_yaws(yaw_npy_paths)
                session_bins_dict = self.find_session_bins(yaw_dict)
                if len(session_bins_dict)!=0:
                    self.sequence_session_bins_frames_dict [(sequence , str(session).split('/')[-1])] = session_bins_dict

    # Preprocess the yaws when yaw_method == 'close_uniform'
    # This function goes over all the sequences to find bins in each session
    # bins_sequence_session_frames_dict [bin] = (sequence, session) pairs that contain frames in the bin
    def close_uniform_preprocess (self):
        self.bins_sequence_session_frames_dict = {}
        for current_bin in self.bins:
            sequence_session_frames_dict = {}
            temp_sequences = []
            for sequence in self.sequences:
                sessions = sorted(list(pathlib.Path(self.yaw_dir + '/' + sequence).glob('*')))
                for session in sessions:
                    yaw_npy_paths = sorted(list(pathlib.Path(str(session)).glob('*')))
                    yaw_dict = self.load_session_yaws(yaw_npy_paths)
                    frames_for_bin = self.find_frames_for_bin(yaw_dict, current_bin)
                    if len (frames_for_bin)>0:
                        if sequence not in temp_sequences:
                            temp_sequences.append(sequence)
                        sequence_session_frames_dict [(sequence , str(session).split('/')[-1])] = frames_for_bin
            
            if len(temp_sequences)!=0:            
                self.bins_sequence_session_frames_dict[(str(current_bin),0)] = temp_sequences
                self.bins_sequence_session_frames_dict[(str(current_bin),1)] = sequence_session_frames_dict
        # Existing bins in the dataset 
        self.possible_bins = [k[0] for k in self.bins_sequence_session_frames_dict.keys() if k[1]==0]
        print(self.phase, "possible bins", self.possible_bins)
        

    def load_npy(self, path_string):
        np_array = np.load(path_string)
        yaw = np_array [0]
        return yaw
    
    # Make a dictionary yaw_dict [frame_number] = frame's_yaw_angle 
    def load_session_yaws (self, yaw_npy_paths):
        yaw_dict = {}
        for yaw_path in yaw_npy_paths:
            frame_num = (str(yaw_path).split('/')[-1]).split('.')[0]
            yaw_dict [frame_num] = self.load_npy(yaw_path)
        return yaw_dict

    # Outputs the keys in a dictionary where the value is in a certain range
    def find_min_max_poses (self, yaw_dict):
        return [k for k,v in yaw_dict.items() if (np.abs(v)>= self.abs_min_yaw and np.abs(v)<= self.abs_max_yaw)]
    
    # Outputs the bin_dict[current_bin] = frame_numbers in the session that belong to the current_bin
    def find_session_bins (self, yaw_dict):
        bin_dict = {}
        for current_bin in self.bins:
            frames_for_bin = self.find_frames_for_bin (yaw_dict, current_bin)
            if len(frames_for_bin) >= self.args.num_source_frames + self.args.num_target_frames:
                bin_dict[str(current_bin)] = frames_for_bin
        return bin_dict
    
    # Outputs frame_numbers in the yaw_dict that belong to the current_bin
    def find_frames_for_bin (self, yaw_dict, current_bin):
        frames_for_bin = []
        value = [k for k,v in yaw_dict.items() if (v>= current_bin[0] and v<= current_bin[1])]
        if len(value) >= self.args.num_source_frames + self.args.num_target_frames:
            frames_for_bin = value
        return frames_for_bin

    # Get the existing sessions for a sequence in the keys belonging to sequence_session_dict 
    def get_sessions (self, sequence , sequence_session_dict):
        sequence_session = sequence_session_dict.keys()
        return [v for k,v in sequence_session if k == sequence]
    

    def __getitem__(self, index):

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
        
        else:
            # Sample source and target frames for the current video sequence
            filenames = []
            difficult_frames = []

            if self.yaw_method == 'min_max' :
                sessions = self.get_sessions(self.sequences[index], self.sequence_session_frames_dict)
                # Sample one session from the video
                random_session = random.sample(sessions, 1)[0]
                difficult_frames = self.sequence_session_frames_dict [(self.sequences[index], random_session)]

            if self.yaw_method == 'close_original' :
                sessions = self.get_sessions(self.sequences[index], self.sequence_session_bins_frames_dict)
                # Sample one session from the video
                random_session = random.sample(sessions, 1)[0]
                session_dict = self.sequence_session_bins_frames_dict[(self.sequences[index], random_session)]
                session_bins = session_dict.keys()
                random_bin = random.sample(session_bins, 1)[0]
                difficult_frames = session_dict[random_bin]


            if self.yaw_method == 'close_uniform' :
                self.sequence_session_frames_dict = self.bins_sequence_session_frames_dict[(str(self.current_bin),1)]
                sessions = self.get_sessions(self.sequences[index], self.sequence_session_frames_dict)
                # Sample one session from the video
                random_session = random.sample(sessions, 1)[0]
                difficult_frames = self.sequence_session_frames_dict [(self.sequences[index], random_session)]


            # The source and target relative paths (sample one pair from the close keypoints in a session)
            # source_target_pair is a randomly selected pair of two possible frame number to be chosen
            # filenames will be the absolute path to these two frame numbers. The source and target pair will
            # randomly be chosen from filenames
            source_target_pair = random.sample(difficult_frames, 2)
            filenames = [pathlib.Path(self.sequences[index]+'/'+random_session+'/'+source_target_pair[0]),
                        pathlib.Path(self.sequences[index]+'/'+random_session+'/'+source_target_pair[1])]
            
            if self.args.same_source_and_target:
                filenames = [pathlib.Path(self.sequences[index]+'/'+random_session+'/'+source_target_pair[0]),
                            pathlib.Path(self.sequences[index]+'/'+random_session+'/'+source_target_pair[0])]
            
            random.shuffle(filenames)

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
            keypoints = keypoints.reshape((68,2)) #I added this
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

                # Convert 3-channel segmentations to 1 grayscale image segs += [self.to_tensor(seg)]
                segs += [self.to_tensor(seg)[0][None]]

            sample_from_reserve = False

        # Normalized the images and poses in (-1,1) range
        imgs = (torch.stack(imgs)- 0.5) * 2.0
        poses = (torch.stack(poses) - 0.5) * 2.0

        if self.args.output_stickmen:
            stickmen = ds_utils.draw_stickmen(self.args, poses)

        if self.args.output_segmentation:
            segs = torch.stack(segs)

        # Split between few-shot source and target sets
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

        # When the yaw_method is close_uniform, first the bin should be selected 
        # When all the pairs are selected from the videos, select a random bin for the next round
        if index == len(self.sequences)-1 and self.yaw_method == 'close_uniform':
            self.change_bin = True
            self.change_current_bin()

        return data_dict

    def change_current_bin (self):
        if self.change_bin:
            self.current_bin = random.sample(self.possible_bins,1)[0]
            print("Changing the bin: new bin ", self.current_bin)
            self.change_bin = False
            self.sequences = self.bins_sequence_session_frames_dict[(str(self.current_bin),0)]
    
    def __len__(self):
        return len(self.sequences)

    # To shuffle the dataset before the epoch
    def shuffle(self):
        if self.phase != 'metrics': # Don't shuffle metrics
            self.sequences = [self.sequences[i] for i in torch.randperm(len(self.sequences)).tolist()]
