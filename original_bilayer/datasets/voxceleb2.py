import torch
import pickle
from torch.autograd import Variable
import dlib
import cv2
from torch.utils import data
from torchvision import transforms
import torchvision
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
import torch.nn.functional as F


class DatasetWrapper(data.Dataset):
    @staticmethod
    def get_args(parser):
        # Common properties
        parser.add('--experiment_dir',           default='.', type=str,
                                                 help='directory to save logs')

        parser.add('--experiment_name',          default='test', type=str,
                                                 help='name of the experiment used for logging')

        parser.add('--bin_path',                 default='.', type=str,
                                                 help='has the files for each angle')

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
        
        parser.add('--cutoff_shirt',             default='False', type=rn_utils.str2bool, choices=[True, False],
                                                 help='If True, cuts off shirt in segmentation')
        
        parser.add('--save_dataset_filenames',   default='False', type=rn_utils.str2bool, choices=[True, False],
                                                 help='If True, the train/test data is saved in train/test_filnames.txt')

        parser.add('--train_load_from_filename', default='train_filnames.txt', type=str,
                                                 help='filename that we read the training dataset images from if dataset_load_from_txt==True')                                    

        parser.add('--test_load_from_filename',  default='test_filnames.txt', type=str,
                                                 help='filename that we read the testing dataset images from if dataset_load_from_txt==True')

        parser.add('--augment_with_general',     default='False', type=rn_utils.str2bool, choices=[True, False],
                                                 help='gradually increase the weight of general dataset while training the per_person dataset')

        parser.add('--mask_source_and_target',   default='True', type=rn_utils.str2bool, choices=[True, False],
                                                 help='mask the source and target from the beginning')
        
        parser.add('--same_source_and_target',   default='False', type=rn_utils.str2bool, choices=[True, False],
                                                 help='set source = target in the experiment')
                                                      
        parser.add('--rebalance',                default='False', type=rn_utils.str2bool, choices=[True, False],
                                                 help='rebalance the dataset?')

        return parser

    def __init__(self, args, phase, pose_component = 'none'):
        super(DatasetWrapper, self).__init__()
        # Store options
        self.phase = phase
        self.args = args
        self.train_load_index = 0
        self.test_load_index = 0

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

            # choosing the general dataset root to sample from in training per-person approach
            # if self.args.augment_with_general and args.data_root != args.general_data_root:
            #     general_data_root = args.general_data_root
        else:
            data_root = args.data_root

        if phase == 'metrics':
            data_root = args.metrics_root

        if phase != 'metrics' and args.rebalance:
            print('Loaded bins\n')
            self.bins = pickle.load(open(args.bin_path, "rb"))

        # Data paths
        self.imgs_dir = pathlib.Path(data_root) / 'imgs' / phase
        self.pose_dir = pathlib.Path(data_root) / 'keypoints' / phase

        if args.output_segmentation:
            self.segs_dir = pathlib.Path(data_root) / 'segs' / phase

        # Video sequences list
        sequences = self.imgs_dir.glob('*/*')
        self.sequences = ['/'.join(str(seq).split('/')[-2:])
                          for seq in sequences]

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

        # make a directory to save test and train paths
        # Prepare experiment directories and save options
        experiment_dir = pathlib.Path(args.experiment_dir)
        self.experiment_dir = experiment_dir / 'runs' / args.experiment_name

        # if self.args.dataset_load_from_txt:
        #     self.args.save_dataset_filenames = False
        # ===============================================================

        print('Loading data.')

    def __getitem__(self, index):
        # Sample source and target frames for the current sequence
        count = 0
        filenames = []

        while True:
            count += 1
            try:
                filenames_img = list((self.imgs_dir / self.sequences[index]).glob('*/*'))
                filenames_img = [pathlib.Path(
                    *filename.parts[-4:]).with_suffix('') for filename in filenames_img]

                filenames_npy = list((self.pose_dir / self.sequences[index]).glob('*/*'))
                filenames_npy = [pathlib.Path(
                    *filename.parts[-4:]).with_suffix('') for filename in filenames_npy]

                filenames = list(
                    set(filenames_img).intersection(set(filenames_npy)))

                if self.args.output_segmentation:
                    filenames_seg = list(
                        (self.segs_dir / self.sequences[index]).glob('*/*'))
                    filenames_seg = [pathlib.Path(
                        *filename.parts[-4:]).with_suffix('') for filename in filenames_seg]

                    filenames = list(
                        set(filenames).intersection(set(filenames_seg)))

                if len(filenames) != 0:
                    break
                else:
                    raise  # the length of filenames is zero.

            except Exception as e:
                print("# Exception is raised if filenames list is empty or there was an error during read")
                index = (index + 1) % len(self)

        filenames = sorted(filenames)
        
        if self.args.same_source_and_target:
            selected_filename = random.randint(0, (len(filenames) - 1))
            filenames = [filenames[selected_filename] for i in filenames]

        imgs = []
        poses = []
        stickmen = []
        segs = []

        reserve_index = -1  # take this element of the sequence if loading fails
        sample_from_reserve = False

        if self.phase == 'test':
            # Sample from the beginning of the sequence
            self.cur_num = 0

        while len(imgs) < self.args.num_source_frames + self.args.num_target_frames:
            if reserve_index == len(filenames):
                raise  # each element of the filenames list is unavailable for load

            # Sample a frame number
            if sample_from_reserve:
                filename = filenames[reserve_index]

            else:

                if self.phase == 'metrics':
                    # If you are taking the metrics, you want to return frame_num 0 then frame_num 1
                    # we can check which one to return in this iteration by checking the length of imgs
                    frame_num = len(imgs)
                    filename = filenames[frame_num]
                elif self.args.frame_num_from_paper:
                    frame_num = int(round(self.cur_num * (len(filenames) - 1)))
                    self.cur_num = (self.cur_num + self.delta) % 1

                    filename = filenames[frame_num]
                # Samples from a constant distribution of poses by selecting from pose bins
                elif self.args.rebalance:
                    # Get the correct video uid
                    video_uid = filenames[0].parent.parent.name
                    bins = self.bins[video_uid]

                    # Remove zero sized bins
                    bins = [i for i in bins if len(i) != 0]

                    # Get a random bin
                    bin_index = torch.randint(0, len(bins), (1,))

                    # Get a frame within that bin
                    frame_num = torch.randint(0, len(bins[bin_index]), (1,))
                    filename_raw = bins[bin_index][frame_num]

                    # Turn that into a usable value
                    files_split = filename_raw.split('/')
                    filename_cleaned = files_split[-4:-1] + [files_split[-1][:-4]]
                    filename_cleaned = '/'.join(filename_cleaned)
                    filename = pathlib.Path(filename_cleaned)
                else:
                    frame_num = random.randint(0, (len(filenames) - 1))
                    filename = filenames[frame_num]

            # Read images
            img_path = pathlib.Path(self.imgs_dir) / filename.with_suffix('.jpg')

            if self.phase == 'test':
                print("selected test image path is: ", img_path)

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
                    if len(imgs) <= self.args.num_source_frames:
                        save_file = self.phase + "_filenames.txt"
                        with open(self.experiment_dir / save_file, 'a') as data_file:
                            data_file.write('source %s:%s\n' % (
                                str(len(imgs)), str(filename.with_suffix('.jpg'))))

                    if len(imgs) > self.args.num_source_frames:
                        save_file = self.phase + "_filenames.txt"
                        with open(self.experiment_dir / save_file, 'a') as data_file:
                            data_file.write('target %s:%s\n' % (str(len(imgs)-self.args.num_source_frames),
                                                                str(filename.with_suffix('.jpg'))))

            except:
                imgs.pop(-1)

                sample_from_reserve = True
                reserve_index += 1
                continue

            keypoints = keypoints.reshape((68, 2))  # I added this
            keypoints = keypoints[:self.args.num_keypoints, :]
            boundary = int(np.max(keypoints[:, 1]))
            keypoints[:, :2] /= s
            keypoints = keypoints[:, :2]
            update_seg = torch.ones(1, 256, 256)
            if self.args.cutoff_shirt:
                # Cuts off everything below the bottom most keypoint.
                # When combined with the segmentation mask which cuts off all the background, this
                # leaves only the face part remaining in the image
                update_seg[:, boundary:, :] = 0
            poses += [torch.from_numpy(keypoints.reshape(-1))]

            if self.args.output_segmentation:
                seg_path = pathlib.Path(self.segs_dir) / \
                    filename.with_suffix('.png')

                try:
                    seg = Image.open(seg_path)
                    seg = seg.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
                except:
                    imgs.pop(-1)
                    poses.pop(-1)

                    sample_from_reserve = True
                    reserve_index += 1
                    continue

                # Weird segmentation fix to change RGB into b/w
                segs += [self.to_tensor(seg)[0][None]*update_seg]
            
            sample_from_reserve = False

        imgs = (torch.stack(imgs) - 0.5) * 2.0

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

        return data_dict

    def __len__(self):
        return len(self.sequences)

    def shuffle(self):
        if self.phase != 'metrics':  # Don't shuffle metrics
            self.sequences = [self.sequences[i]
                              for i in torch.randperm(len(self.sequences)).tolist()]
