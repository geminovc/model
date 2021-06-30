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

        # Yaws                              
        parser.add('--root_to_yaws',             default='/data/pantea/pose_results/yaws/per_person_1_three_datasets/angles', type=str, 
                                                 help='The directory where the yaws are stored in voxceleb2 format')
        
        parser.add('--abs_min_yaw',              default=45, type=float, 
                                                 help='The minimum abs value for yaw')
        
        parser.add('--abs_max_yaw',              default=90, type=float, 
                                                 help='The maximum abs value for yaw') 
        
        parser.add('--mask_source_and_target',   default='True', type=rn_utils.str2bool, choices=[True, False],
                                                 help='Mask the souce and target from the beginning')
        
        parser.add('--yaw_method',               default='close_original', type=str, 
                                                 help='The method by which the soucre-target pair is slected based on yaw. Possible choices: min_max, close_uniform, close_original')                                                      
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
            #data_root = (data_list[0].split(":"))[1]            
            
        else:
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
        

        # Parameters of the sampling scheme
        self.delta = math.sqrt(5)
        self.cur_num = torch.rand(1).item()

        #make a directory to save test and train paths
        # Prepare experiment directories and save options
        experiment_dir = pathlib.Path(args.experiment_dir)
        self.experiment_dir = experiment_dir / 'runs' / args.experiment_name

        if self.args.yaw_method == 'min_max':
            # We did no have enough difficult pose in the test
            if self.phase == 'test':
                self.args.abs_min_yaw = 30

            self.min_max_preprocess ()
        
        if self.args.yaw_method == 'close_original':
            # self.bins = [[-5,5],[5,15],[15,25],[25,35],[35,45],[45,55],[55,65][]]
            self.bins = [[-90,-75],[-75,-45],[-45,-15],[-15,15],[15,45],[45,75],[75,90]]
            self.close_original_preprocess ()
        
        if self.args.yaw_method == 'close_uniform':
            # self.bins = [[-5,5],[5,15],[15,25],[25,35],[35,45],[45,55],[55,65][]]
            self.bins = [[-90,-45],[-45,0],[0,45],[45,90]]
            self.close_uniform_preprocess ()
            self.change_bin = True
            self.change_current_bin()


    def min_max_preprocess (self):
        print("length of sequences before removing easy yaws",len(self.sequences))
        print("sequences before removing easy yaws",self.sequences)
        temp_sequences = []
        self.sequence_session_frames_dict = {}
        # This function goes over all the sequences to delete the ones that don't contain yaws in the desired range
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
                print(sequence, "does not have difficult yaws.")
        
        self.sequences = temp_sequences
        print("length of sequences after removing easy yaws", len(self.sequences))
        print("sequences after removing easy yaws", self.sequences)

    def close_original_preprocess (self):

        self.sequence_session_bins_frames_dict = {}
        # This function goes over all the sequences to find bins in eacch session
        for sequence in self.sequences:
            sessions = sorted(list(pathlib.Path(self.yaw_dir + '/' + sequence).glob('*')))
            for session in sessions:
                yaw_npy_paths = sorted(list(pathlib.Path(str(session)).glob('*')))
                yaw_dict = self.load_session_yaws(yaw_npy_paths)
                session_bins_dict = self.find_session_bins(yaw_dict)
                self.sequence_session_bins_frames_dict [(sequence , str(session).split('/')[-1])] = session_bins_dict

    def close_uniform_preprocess (self):
        self.bins_sequence_session_frames_dict = {}
        # This function goes over all the sequences to find bins in each session
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
        

        self.possible_bins = [k[0] for k in self.bins_sequence_session_frames_dict.keys() if k[1]==0]
        print(self.phase, "possible bins", self.possible_bins)
        

    def load_npy(self, path_string):
        np_array = np.load(path_string)
        yaw = np_array [0]
        return yaw
    
    def load_session_yaws (self, yaw_npy_paths):
        yaw_dict = {}
        for yaw_path in yaw_npy_paths:
            frame_num = (str(yaw_path).split('/')[-1]).split('.')[0]
            yaw_dict [frame_num] = self.load_npy(yaw_path)
        return yaw_dict

    def find_min_max_poses (self, yaw_dict):
        return [k for k,v in yaw_dict.items() if (np.abs(v)>= self.args.abs_min_yaw and np.abs(v)<= self.args.abs_max_yaw)]
    
    def find_session_bins (self, yaw_dict):
        bin_dict = {}
        for current_bin in self.bins:
            value = [k for k,v in yaw_dict.items() if (v>= current_bin[0] and v<= current_bin[1])]
            if len(value) >= self.args.num_source_frames + self.args.num_target_frames:
                bin_dict[str(current_bin)] = value
        return bin_dict

    def find_frames_for_bin (self, yaw_dict, current_bin):
        frames_for_bin = []
        value = [k for k,v in yaw_dict.items() if (v>= current_bin[0] and v<= current_bin[1])]
        if len(value) >= self.args.num_source_frames + self.args.num_target_frames:
            frames_for_bin = value
        return frames_for_bin

    def get_sessions (self, sequence , sequence_session_dict):
        sequence_session = sequence_session_dict.keys()
        return [v for k,v in sequence_session if k == sequence]
    

    def __getitem__(self, index):
        
        # Sample source and target frames for the current video sequence
        filenames = []
        difficult_frames = []

        if self.args.yaw_method == 'min_max' :
            sessions = self.get_sessions(self.sequences[index], self.sequence_session_frames_dict)
            # Sample one session from the video
            random_session = random.sample(sessions, 1)[0]
            difficult_frames = self.sequence_session_frames_dict [(self.sequences[index], random_session)]


        if self.args.yaw_method == 'close_original' :
            sessions = self.get_sessions(self.sequences[index], self.sequence_session_bins_frames_dict)
            # Sample one session from the video
            random_session = random.sample(sessions, 1)[0]
            session_dict = self.sequence_session_bins_frames_dict[(self.sequences[index], random_session)]
            session_bins = session_dict.keys()
            random_bin = random.sample(session_bins, 1)[0]
            difficult_frames = session_dict[random_bin]


        if self.args.yaw_method == 'close_uniform' :
            self.sequence_session_frames_dict = self.bins_sequence_session_frames_dict[(str(self.current_bin),1)]
            sessions = self.get_sessions(self.sequences[index], self.sequence_session_frames_dict)
            # Sample one session from the video
            random_session = random.sample(sessions, 1)[0]
            difficult_frames = self.sequence_session_frames_dict [(self.sequences[index], random_session)]


        # The source and target relative paths (sample one pair from the close keypoints in a session)
        source_target_pair = random.sample(difficult_frames, 2)
        filenames = [pathlib.Path(self.sequences[index]+'/'+random_session+'/'+source_target_pair[0]),
                     pathlib.Path(self.sequences[index]+'/'+random_session+'/'+source_target_pair[1])]
        
        random.shuffle(filenames)
        print("filenames",filenames)

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
        
        if self.args.mask_source_and_target and self.args.output_segmentation:
            data_dict['source_imgs'] = data_dict['source_imgs'] * data_dict['source_segs'] + (-1) * (1 - data_dict['source_segs'])
            data_dict['target_imgs'] = data_dict['target_imgs'] * data_dict['target_segs'] + (-1) * (1 - data_dict['target_segs'])      
                
        data_dict['indices'] = torch.LongTensor([index])

        if index == len(self.sequences)-1 and self.args.yaw_method == 'close_uniform':
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

    def shuffle(self):
        if self.phase != 'metrics': # Don't shuffle metrics
            self.sequences = [self.sequences[i] for i in torch.randperm(len(self.sequences)).tolist()]
