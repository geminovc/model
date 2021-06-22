"""
This file is for picking the source and target frames that have keypoints with L2 disance less than a threshold.
Before using this dataloader, you need to do a preprocess on the dataset using the difficult_poses module.

The structure of input dataset is:

ROOT_TO_CLOSE_KEYPOINTS/[train, test]/PERSON_ID/VIDEO_ID/SEQUENCE_ID/L2_Distances.pkl

Each L2_Distances.pkl file contains a dictionary with the following structure:

{
('-1','-1'): 'data_root of the dataset that the keypoints belong to',
('frame_num_1','frame_num_1'): 'L2_distance (keypoint_1, keypoint_2)',
...
}

Example of the close_keypoints structure:

                     ROOT_TO_CLOSE_KEYPOINTS _ phase _ id00012 _ abc _ 00001 _ L2_Distances.pkl
                                                    |         |            
                                                    |         |           
                                                    |         |            
                                                    |         |
                                                    |         |_ def  _ 00001 _ L2_Distances.pkl
                                                    |                |       
                                                    |                |     
                                                    |                |       
                                                    |                |
                                                    |                |_ 00002 _ L2_Distances.pkl
                                                    |                        
                                                    |                      
                                                    |                        
                                                    |               
                                                    |_ id00013 _ lmn _ 00001 _ L2_Distances.pkl
                                                    |          |            
                                                    |          |       
                                                    |          |           
                                                    |          |
                                                    |          |_ opq  _ 00001 _ ...
                                                    |                 |_ 00002 _ ...
                                                    |                 |_ 00003 _ ...
                                                    |
                                                    |_ id00014 _ rst _ 00001 _ ...
                                                                |    |_ 00002 _ ...
                                                                |
                                                                |_ uvw  _ 00001 _ L2_Distances.pkl
                                                                        |      
                                                                        |     
                                                                        |      
                                                                        |
                                                                        |_ 00002 _ L2_Distances.pkl
                                                                        |     
                                                                        |     
                                                                        |
                                                                        |_ 00003 _L2_Distances.pkl

Note that the last folder of ROOT_TO_CLOSE_KEYPOINTS has the same name as DATA_ROOT, meaning that the L2 distances of ROOT_TO_CLOSE_KEYPOINTS belong to datset in DATA_ROOT. 

Example:
ROOT_TO_CLOSE_KEYPOINTS = /data/pantea/close_keypoints/per_person_extracts
DATA_ROOT = /video-conf/scratch/pantea/per_person_extracts

In find_l2_distance module we have:

ROOT_TO_CLOSE_KEYPOINTS = RESULTS_FOLDER/DATASET_NAME

Arguments
----------

root_to_close_keypoints: The root directory to where the L2_distances of the input data_root is stored. 
        
close_keypoints_threshold: The threshold with which we call two keypoints close. 



Outputs
----------

The output is data_dict which contains source and target images with L2 keypoint distance of less than close_keypoints_threshold. It contains: 
'target_stickmen'  
'source_stickmen'  
'source_imgs'  
'target_imgs'  
'source_segs'  
'target_segs'  

The script is failful to the papers implementation and it picks one random source/target pair, with keyponits having L2 distance of less than close_keypoints_threshold, from each video.

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
        
        parser.add('--frame_num_from_paper',     default='False', type=rn_utils.str2bool, choices=[True, False],
                                                 help='The random method to sample frame numbers for source and target from dataset')
        
        parser.add('--dataset_load_from_txt',    default='False', type=rn_utils.str2bool, choices=[True, False],
                                                 help='If True, the train is loaded from train_load_from_filename, the test is loaded from test_load_from_filename. If false, the data is loaded from data-root')
        
        parser.add('--save_dataset_filenames',   default='False', type=rn_utils.str2bool, choices=[True, False],
                                                 help='If True, the train/test data is saved in train/test_filnames.txt')

        parser.add('--train_load_from_filename', default='train_filnames.txt', type=str,
                                                 help='filename that we read the training dataset images from if dataset_load_from_txt==True')                                    

        parser.add('--test_load_from_filename',  default='test_filnames.txt', type=str,
                                                 help='filename that we read the testing dataset images from if dataset_load_from_txt==True')

        parser.add('--augment_with_general',     default='False', type=rn_utils.str2bool, choices=[True, False],
                                                 help='gradually increase the weight of general dataset while training the per_person dataset')

        parser.add('--txt_directory',            default='/data4/pantea/nets_implementation/original_bilayer/difficult_poses/results/temp_per_person_extracts', type=str, 
                                                 help='gradually increase the weight of general dataset while training the per_person dataset')

        # Paired target and source (close keypoints)                              

        parser.add('--root_to_close_keypoints',  default='/data4/pantea/nets_implementation/original_bilayer/difficult_poses/results/temp_per_person_extracts', type=str, 
                                                 help='If True, the images, keypoints, and segs are picked from files')
        
        parser.add('--close_keypoints_threshold',default=500, type=float,
                                                 help='threshold of defining two close keypoints')
        
        return parser

    def __init__(self, args, phase):
        super(DatasetWrapper, self).__init__()
        # Store options
        self.phase = phase
        self.args = args
        self.train_load_index=0
        self.test_load_index=0

        self.to_tensor = transforms.ToTensor()
        self.epoch = 0 if args.which_epoch == 'none' else int(args.which_epoch)


        if self.args.dataset_load_from_txt:

            if self.phase == 'train':
                my_file = open(str(self.args.train_load_from_filename), "r")
            else:
                my_file = open(str(self.args.test_load_from_filename), "r")

            content = my_file.read()
            data_list = content.split("\n")
            my_file.close()
            data_root = args.data_root
            
        else:
            data_root = args.data_root

        if phase == 'metrics':
            data_root = args.metrics_root
        

        self.imgs_dir = pathlib.Path(data_root) / 'imgs' / phase
        self.pose_dir = pathlib.Path(data_root) / 'keypoints' / phase

        if args.output_segmentation:
            self.segs_dir = pathlib.Path(data_root) / 'segs' / phase
        
        self.close_keypoints_dir = self.args.root_to_close_keypoints + '/' + self.phase
        
        # Video sequences list
        # Please don't mix the sequence, which is the total number of videos, with sessions (which are essentially short video clips).
        # Each sequence, is made up of multiple sessions.

        sequences = self.imgs_dir.glob('*/*')
        self.sequences = ['/'.join(str(seq).split('/')[-2:]) for seq in sequences]

        # Since the general dataset is too big, we sample 22 items if sample is chosen
        if self.args.sample_general_dataset and self.args.data_root == self.args.general_data_root:
            self.sequences = random.sample(self.sequences, 22)

        # If you are using metrics, sort the Sequences so it's easier
        # to keep track of them between runs
        if phase == 'metrics':
            self.sequences = sorted(self.sequences)
        
        print(len(self.sequences), self.sequences)

        # Parameters of the sampling scheme
        self.delta = math.sqrt(5)
        self.cur_num = torch.rand(1).item()

        #make a directory to save test and train paths
        # Prepare experiment directories and save options
        experiment_dir = pathlib.Path(args.experiment_dir)
        self.experiment_dir = experiment_dir / 'runs' / args.experiment_name

        # if self.args.dataset_load_from_txt:
        #     self.args.save_dataset_filenames = False

    # Load the pickle files as dictionary
    def load_pickle(self, path_string):
        pkl_file = open(path_string, 'rb')
        my_dict = pickle.load(pkl_file)
        pkl_file.close()
        return my_dict

    def find_close_keypoints (self, keypoints_dict, threshold):
        # The ('-1','-1') key contains the data_root of the keypoints
        keypoints_dict.pop(('-1', '-1'), None)
        return [k for k,v in keypoints_dict.items() if float(v) >= threshold and k!= ('-1', '-1')]



    def __getitem__(self, index):
        
        # No close keypoints for metrics loader
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
            keypoints_pickles = list(pathlib.Path(self.close_keypoints_dir + '/' + self.sequences[index]).glob('*/*'))
            
            # Sample one session from the video
            keypoints_pickle_path = random.sample(keypoints_pickles, 1)[0]
            keypoints_dict =  self.load_pickle(keypoints_pickle_path)
            close_keys = self.find_close_keypoints(keypoints_dict, self.args.close_keypoints_threshold)
            
            
            # The source and target relative paths (sample one pair from the close keypoints in a session)
            relative_path = '/'.join(str(keypoints_pickle_path).split('/')[-4:-1])
            source_target_pair = random.sample(close_keys, 1)[0]
            filenames = [pathlib.Path(relative_path+'/'+source_target_pair[0]), pathlib.Path(relative_path+'/'+source_target_pair[1])]
            random.shuffle(filenames)
            print(filenames)

        imgs = []
        poses = []
        stickmen = []
        segs = []

        reserve_index = -1 # take this element of the sequence if loading fails
        sample_from_reserve = False

        if self.phase == 'test':
            # Sample from the beginning of the sequence
            self.cur_num = 0
        
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
            
            if self.phase == 'test':
                print("selected test image path is: ",img_path)

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
                
                if self.args.save_dataset_filenames:
                    # write the filename to file
                    if len (imgs) <= self.args.num_source_frames :
                        save_file = self.phase + "_filenames.txt"
                        with open(self.experiment_dir / save_file, 'a') as data_file:
                            data_file.write('source %s:%s\n' % (str(len (imgs)), str(filename.with_suffix('.jpg'))))
                    
                    if len (imgs) > self.args.num_source_frames:
                        save_file = self.phase + "_filenames.txt"
                        with open(self.experiment_dir / save_file, 'a') as data_file:
                            data_file.write('target %s:%s\n' % (str(len (imgs)-self.args.num_source_frames), \
                                    str(filename.with_suffix('.jpg'))))
            
            except:
                imgs.pop(-1)

                sample_from_reserve = True
                reserve_index += 1
                continue

            keypoints = keypoints.reshape((68,2)) #I added this
            keypoints = keypoints[:self.args.num_keypoints, :]
            keypoints[:, :2] /= s
            keypoints = keypoints[:, :2]


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


                # Convert 3-channel segmentations to 1 grayscale image
                # segs += [self.to_tensor(seg)]
                segs += [self.to_tensor(seg)[0][None]]

            sample_from_reserve = False

        imgs = (torch.stack(imgs)- 0.5) * 2.0

        poses = (torch.stack(poses) - 0.5) * 2.0

        

        if self.args.output_stickmen:
            stickmen = ds_utils.draw_stickmen(self.args, poses)

        if self.args.output_segmentation:
            segs = torch.stack(segs)

        # Split between few-shot source and target sets
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
        
        data_dict['indices'] = torch.LongTensor([index])

        return data_dict

    def __len__(self):
        return len(self.sequences)

    def shuffle(self):
        if self.phase != 'metrics': # Don't shuffle metrics
            self.sequences = [self.sequences[i] for i in torch.randperm(len(self.sequences)).tolist()]
