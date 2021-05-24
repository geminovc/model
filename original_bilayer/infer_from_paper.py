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
import pdb
from datasets import utils as ds_utils
from runners import utils as rn_utils
from external.Graphonomy import wrapper
import face_alignment



class InferenceWrapper(nn.Module):
    @staticmethod
    def get_args(args_dict):
        # Read and parse args of the module being loaded

        if args_dict['experiment_name'] == 'vc2-hq_adrianb_paper_main' or args_dict['experiment_name'] =='vc2-hq_adrianb_paper_enhancer':
            args_path = pathlib.Path(args_dict['experiment_dir']) / 'bilayer_paper_runs' / args_dict['experiment_name'] / 'args.txt'
            args_dict['croped_segmentation'] = True
        else:
            args_path = pathlib.Path(args_dict['experiment_dir']) / 'runs' / args_dict['experiment_name'] / 'args.txt'

        parser = argparse.ArgumentParser(conflict_handler='resolve')
        parser.add = parser.add_argument

        with open(args_path, 'rt') as args_file:
            lines = args_file.readlines()
            for line in lines:
                k, v, v_type = rn_utils.parse_args_line(line)
                parser.add('--%s' % k, type=v_type, default=v)

        args, _ = parser.parse_known_args()

        # Add args from args_dict that overwrite the default ones
        for k, v in args_dict.items():
            setattr(args, k, v)

        args.world_size = args.num_gpus

        return args

    def __init__(self, args_dict):
        super(InferenceWrapper, self).__init__()
        # Get a config for the network
        self.args = self.get_args(args_dict)
        self.to_tensor = transforms.ToTensor()



        self.runner = importlib.import_module(f'runners.{self.args.runner_name}').RunnerWrapper(self.args, training=False)
        #self.runner.eval()

        # Load pretrained weights
        if args_dict['experiment_name'] == 'vc2-hq_adrianb_paper_main' or args_dict['experiment_name'] =='vc2-hq_adrianb_paper_enhancer':
            checkpoints_dir = pathlib.Path(self.args.experiment_dir) / 'bilayer_paper_runs' / self.args.experiment_name / 'checkpoints'
        else:
            checkpoints_dir = pathlib.Path(self.args.experiment_dir) / 'runs' / self.args.experiment_name / 'checkpoints'

        # Load pre-trained weights
        init_networks = rn_utils.parse_str_to_list(self.args.init_networks) if self.args.init_networks else {}
        frozen_networks = rn_utils.parse_str_to_list(self.args.frozen_networks) if self.args.frozen_networks else {}
        networks_to_train = self.runner.nets_names_to_train
        print("init_networks:", init_networks)
        #print("frozen_networks", frozen_networks)
        #print("networks_to_train", networks_to_train)

        if self.args.init_which_epoch != 'none' and self.args.init_experiment_dir:
            for net_name in init_networks:
                print("loaded ", net_name, "from ", str(pathlib.Path(self.args.init_experiment_dir) / 'checkpoints' / f'{self.args.init_which_epoch}_{net_name}.pth'))
                self.runner.nets[net_name].load_state_dict(torch.load(pathlib.Path(self.args.init_experiment_dir) / 'checkpoints' / f'{self.args.init_which_epoch}_{net_name}.pth', map_location='cpu'))

        for net_name in networks_to_train:
            if net_name not in init_networks and net_name in self.runner.nets.keys():
                print("loaded ", net_name, "from ", str(checkpoints_dir / f'{self.args.which_epoch}_{net_name}.pth'))
                self.runner.nets[net_name].load_state_dict(torch.load(checkpoints_dir / f'{self.args.which_epoch}_{net_name}.pth', map_location='cpu'))
        
        # Remove spectral norm to improve the performance
        #self.runner.apply(rn_utils.remove_spectral_norm)

        # Stickman/facemasks drawer
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

        self.net_seg = wrapper.SegmentationWrapper(self.args)

        if self.args.num_gpus > 0:
            self.cuda()

    def change_args(self, args_dict):
        self.args = self.get_args(args_dict)

    def preprocess_data(self, input_imgs, crop_data=True):
        imgs = []
        poses = []
        stickmen = []
        if not self.args.croped_segmentation:
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
                
                if self.args.output_segmentation and not self.args.croped_segmentation:
                    ## here I decided to use the non-croped version of images for segmentations
                    ## The croped version of images would give a segmentation that has a black stripe on top
                    img_segs = Image.fromarray(np.array(input_imgs[i]))
                    img_segs = img_segs.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
                    imgs_segs.append((self.to_tensor(img_segs) - 0.5) * 2)



                # Crop images and poses
                img = Image.fromarray(input_imgs[i])

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

        
        if input_imgs is not None and self.args.output_segmentation and not self.args.croped_segmentation:
            imgs_segs = torch.stack(imgs_segs, 0)[None]
            if self.args.num_gpus > 0:        
                imgs_segs = imgs_segs.cuda()

        segs = None
        if hasattr(self, 'net_seg') and not isinstance(imgs, list):
            if self.args.croped_segmentation:
                segs = self.net_seg(imgs)[None]
            else:
                segs = self.net_seg(imgs_segs)[None]
                

        return poses, imgs, segs, stickmen

    def get_images_from_dataset (self):
        # selected test image path is:  /video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/yi_qz725MjE/00163/1.jpg
        # selected test image path is:  /video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/yi_qz725MjE/00163/66.jpg
        # selected test image path is:  /video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/Z-G8-wqpxwU/00096/74.jpg
        # selected test image path is:  /video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/Z-G8-wqpxwU/00105/26.jpg
        
        # Source Charactristics
        imgs = []
        poses = []
        stickmen = []
        segs = []

        img = Image.open('/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/yi_qz725MjE/00163/1.jpg') # H x W x 3
        #pdb.set_trace()
        # Preprocess an image
        s = img.size[0]
        img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
        imgs += [self.to_tensor(img)]

        keypoints = np.load('/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/keypoints/test/id/yi_qz725MjE/00163/1.npy').astype('float32')
        keypoints = keypoints.reshape((68,2))
        keypoints = keypoints[:self.args.num_keypoints, :]
        keypoints[:, :2] /= s
        keypoints = keypoints[:, :2]
        poses += [torch.from_numpy(keypoints.reshape(-1))]

        seg = Image.open('/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/segs/test/id/yi_qz725MjE/00163/1.png')
        seg = seg.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
        segs += [self.to_tensor(seg)]



        source_imgs = (torch.stack(imgs)- 0.5) * 2.0
        source_poses = (torch.stack(poses) - 0.5) * 2.0
        source_segs = torch.stack(segs)

        if self.args.output_stickmen:
            stickmen = ds_utils.draw_stickmen(self.args, source_poses)
        source_stickmen = stickmen

        imgs = []
        poses = []
        stickmen = []
        segs = []
        print("self.args.output_stickmen: ",self.args.output_stickmen)

        #pdb.set_trace()
        # Target charactristics
        img = Image.open('/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/yi_qz725MjE/00163/66.jpg') # H x W x 3
        # Preprocess an image
        s = img.size[0]
        img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
        imgs += [self.to_tensor(img)]

        keypoints = np.load('/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/keypoints/test/id/yi_qz725MjE/00163/66.npy').astype('float32')
        keypoints = keypoints.reshape((68,2))
        keypoints = keypoints[:self.args.num_keypoints, :]
        keypoints[:, :2] /= s
        keypoints = keypoints[:, :2]
        poses += [torch.from_numpy(keypoints.reshape(-1))]

        seg = Image.open('/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/segs/test/id/yi_qz725MjE/00163/66.png')
        seg = seg.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
        segs += [self.to_tensor(seg)]

        if self.args.output_stickmen:
            stickmen = ds_utils.draw_stickmen(self.args, poses)

        target_imgs = (torch.stack(imgs)- 0.5) * 2.0
        target_poses = (torch.stack(poses) - 0.5) * 2.0
        target_segs = torch.stack(segs)

        if self.args.output_stickmen:
            stickmen = ds_utils.draw_stickmen(self.args, target_poses)
        target_stickmen = stickmen


        data_dict = {}

        data_dict['source_imgs'] = source_imgs.unsqueeze(1)
        data_dict['target_imgs'] = target_imgs.unsqueeze(1)
        
    
        data_dict['source_poses'] = source_poses.unsqueeze(1)
        data_dict['target_poses'] = target_poses.unsqueeze(1)


        data_dict['source_segs'] = source_segs.unsqueeze(1)
        data_dict['target_segs'] = target_segs.unsqueeze(1)

        if source_stickmen is not None:
            data_dict['source_stickmen'] = source_stickmen.unsqueeze(1)

        if target_stickmen is not None:
            data_dict['target_stickmen'] = target_stickmen.unsqueeze(1)

        return data_dict
        


    def my_preprocess_data(self, input_imgs, crop_data=True):
        imgs = []
        poses = []
        stickmen = []

        #pdb.set_trace()
        pose = self.fa.get_landmarks(input_imgs)[0]

        center = ((pose.min(0) + pose.max(0)) / 2).round().astype(int)
        size = int(max(pose[:, 0].max() - pose[:, 0].min(), pose[:, 1].max() - pose[:, 1].min()))
        center[1] -= size // 6

        if input_imgs is None:
            # Crop poses
            if crop_data:
                s = size * 2

        else:
            # Crop images and poses
            img = Image.fromarray(input_imgs)

            if crop_data:
                img = img.crop((center[0]-size, center[1]-size, center[0]+size, center[1]+size))
                s = img.size[0]

            img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)

            imgs.append((self.to_tensor(img) - 0.5) * 2)



        poses.append(np.reshape(pose,-1))

        # if self.args.output_stickmen:
        #     stickmen = ds_utils.draw_stickmen(self.args, poses[0])
        #     stickmen = stickmen[None]

        if input_imgs is not None:
            imgs.append(input_imgs)

        
        segs = None
        #if self.args.output_segmentation:
        #    segs = self.net_seg(imgs)[None]

        return np.array(poses), imgs, segs, stickmen

  

    def get_images_from_path (self, source_path = '/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/yi_qz725MjE/00163/1.jpg', target_path = '/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/yi_qz725MjE/00163/66.jpg'):
        # selected test image path is:  /video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/yi_qz725MjE/00163/1.jpg
        # selected test image path is:  /video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/yi_qz725MjE/00163/66.jpg
        # selected test image path is:  /video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/Z-G8-wqpxwU/00096/74.jpg
        # selected test image path is:  /video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/imgs/test/id/Z-G8-wqpxwU/00105/26.jpg
        
        # Source Charactristics
        imgs = []
        poses = []
        stickmen = []
        segs = []

        img = Image.open(source_path) # H x W x 3
        img_array = np.array(img)
        poses_s, imgs_s, segs_s, stickmen_s = self.my_preprocess_data(img_array, crop_data=True)
        #pdb.set_trace()
        # Preprocess an image
        s = img.size[0]
        img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
        imgs += [self.to_tensor(img)]

        keypoints = poses_s
        keypoints = keypoints.reshape((68,2))
        keypoints = keypoints[:self.args.num_keypoints, :]
        keypoints[:, :2] /= s
        keypoints = keypoints[:, :2]
        poses += [torch.from_numpy(keypoints.reshape(-1))]

        pdb.set_trace()

        if self.args.output_segmentation:
            seg = self.net_seg(img_array)[None]
            seg = seg.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
            segs += [self.to_tensor(seg)]



        source_imgs = (torch.stack(imgs)- 0.5) * 2.0
        source_poses = (torch.stack(poses) - 0.5) * 2.0
        source_segs = torch.stack(segs)

        if self.args.output_stickmen:
            stickmen = ds_utils.draw_stickmen(self.args, source_poses)
        source_stickmen = stickmen

        imgs = []
        poses = []
        stickmen = []
        segs = []
        print("self.args.output_stickmen: ",self.args.output_stickmen)

        #pdb.set_trace()
        # Target charactristics
        img = Image.open(target_path) # H x W x 3
        # Preprocess an image
        s = img.size[0]
        img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
        imgs += [self.to_tensor(img)]

        keypoints = np.load('/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/keypoints/test/id/yi_qz725MjE/00163/66.npy').astype('float32')
        keypoints = keypoints.reshape((68,2))
        keypoints = keypoints[:self.args.num_keypoints, :]
        keypoints[:, :2] /= s
        keypoints = keypoints[:, :2]
        poses += [torch.from_numpy(keypoints.reshape(-1))]

        seg = Image.open('/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset/segs/test/id/yi_qz725MjE/00163/66.png')
        seg = seg.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
        segs += [self.to_tensor(seg)]

        if self.args.output_stickmen:
            stickmen = ds_utils.draw_stickmen(self.args, poses)

        target_imgs = (torch.stack(imgs)- 0.5) * 2.0
        target_poses = (torch.stack(poses) - 0.5) * 2.0
        target_segs = torch.stack(segs)

        if self.args.output_stickmen:
            stickmen = ds_utils.draw_stickmen(self.args, target_poses)
        target_stickmen = stickmen


        data_dict = {}

        data_dict['source_imgs'] = source_imgs.unsqueeze(1)
        data_dict['target_imgs'] = target_imgs.unsqueeze(1)
        
    
        data_dict['source_poses'] = source_poses.unsqueeze(1)
        data_dict['target_poses'] = target_poses.unsqueeze(1)


        data_dict['source_segs'] = source_segs.unsqueeze(1)
        data_dict['target_segs'] = target_segs.unsqueeze(1)

        if source_stickmen is not None:
            data_dict['source_stickmen'] = source_stickmen.unsqueeze(1)

        if target_stickmen is not None:
            data_dict['target_stickmen'] = target_stickmen.unsqueeze(1)

        return data_dict
        


    def forward(self, data_dict, crop_data=True, no_grad=True , preprocess = False):
        if 'target_imgs' not in data_dict.keys():
            data_dict['target_imgs'] = None

        if preprocess:
            # Inference without finetuning
            (source_poses, 
            source_imgs, 
            source_segs, 
            source_stickmen) = self.preprocess_data(data_dict['source_imgs'], crop_data)

            (target_poses,
            target_imgs, 
            target_segs, 
            target_stickmen) = self.preprocess_data(data_dict['target_imgs'], crop_data)

            data_dict = {
                'source_imgs': source_imgs,
                'source_poses': source_poses,
                'target_poses': target_poses}

            if len(target_imgs):
                data_dict['target_imgs'] = target_imgs

            if source_segs is not None:
                data_dict['source_segs'] = source_segs

            if target_segs is not None:
                data_dict['target_segs'] = target_segs

            if source_stickmen is not None:
                data_dict['source_stickmen'] = source_stickmen

            if target_stickmen is not None:
                data_dict['target_stickmen'] = target_stickmen

        else:
            
            data_dict = self.get_images_from_path ()


        # # Calculate "standing" stats for the batch normalization
        print("The data_root is:", self.args.data_root)
        train_dataloader = ds_utils.get_dataloader(self.args, 'train')
        train_dataloader.dataset.shuffle()

        if self.args.calc_stats:
            print("Calculate standing stats for the batch normalization")
            self.runner.calculate_batchnorm_stats(train_dataloader, self.args.debug)

        model = self.runner
        # Test
        model.eval()

        # # test_dataloader.dataset.shuffle()
        #for data_dict_idx in data_dict:
        # Prepare input data
        if self.args.num_gpus > 0:
            for key, value in data_dict.items():
                data_dict[key] = value.cuda()

        # Forward pass
        with torch.no_grad():
            model(data_dict)
        
        # if args.debug:
        #     break       


        return model.data_dict