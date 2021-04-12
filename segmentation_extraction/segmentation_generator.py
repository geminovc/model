
import argparse
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import argparse
import torchvision
import os
import sys
sys.path.append("/home/pantea/video-conf/pantea/bilayer-model")
import pathlib
import numpy as np
import cv2
import importlib
import ssl

from datasets import utils as ds_utils
from runners import utils as rn_utils
from external.Graphonomy import wrapper
import face_alignment


class Segmentation_Generator():
    @staticmethod
    def get_args(parser):
        # Common properties
        parser.add('--project_dir',              default='/data/pantea', type=str,
                                                 help='root directory of the code')
        parser.add('--data_root',                default="/data/pantea/video_conf", type=str,
                                                 help='root directory of the train data')
        parser.add('--segs_dir',                 default="/data/pantea/video_conf/", type=str,
                                                 help='root directory of the raw videos')        
        parser.add('--segmentatio_dir',          default="segs", type=str,
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
        parser.add('--num_gpus',            default=1, type=int, 
                                        help='thickness of lines in the stickman')


        # Technical options that are set automatically
        parser.add('--local_rank', default=0, type=int)
        parser.add('--rank',       default=0, type=int)
        parser.add('--world_size', default=1, type=int)
        parser.add('--train_size', default=1, type=int)

        # Dataset options
        args, _ = parser.parse_known_args()

        return parser

    def __init__(self, args, phase):        
        # Store options
        self.phase = phase
        self.args = args


        self.to_tensor = transforms.ToTensor()

        data_root = self.args.data_root
        
        # Data paths
        self.imgs_dir = pathlib.Path(data_root) / 'imgs' / phase

        print("imgs_dir", self.imgs_dir)

        # Video sequences list
        sequences = self.imgs_dir.glob('*/*')
        self.sequences = ['/'.join(str(seq).split('/')[-2:]) for seq in sequences]
        
        self.segs_dir = pathlib.Path(data_root) / 'segs' / phase
        
        self.to_tensor = transforms.ToTensor()


        self.net_seg = wrapper.SegmentationWrapper(self.args)


    def preprocess_data(self, input_imgs, crop_data=True):
        imgs = []

        if len(input_imgs.shape) == 3:
            input_imgs = input_imgs[None]
            N = 1

        else:
            N = input_imgs.shape[0]

        for i in range(N):

            if input_imgs is None:
                # Crop poses
                if crop_data:
                    s = size * 2

            else:
                # Crop images and poses
                img = Image.fromarray(input_imgs[i])
 
                img = img.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)

                imgs.append((self.to_tensor(img) - 0.5) * 2)

        if input_imgs is not None:
            imgs = torch.stack(imgs, 0)[None]

        if self.args.num_gpus > 0:
            
            if input_imgs is not None:
                imgs = imgs.cuda()
    
        segs = None
        if hasattr(self, 'net_seg') and not isinstance(imgs, list):
            segs = self.net_seg(imgs)[None]

        return segs 

    def get_segs (self):
         # Sample source and target frames for the current sequence
        for index in range(0,len(self.sequences)):
            print("Sequences is: ", str(self.sequences[index]))
            filenames_vid = list((self.imgs_dir / self.sequences[index]).glob('*/*'))
            filenames_vid = [pathlib.Path(*filename.parts[-4:]).with_suffix('') for filename in filenames_vid]
            filenames = list(set(filenames_vid))
            filenames = sorted(filenames)
            for filename in filenames:
                #print("filename is:", str(filename))
                img_path = pathlib.Path(self.imgs_dir) / filename.with_suffix('.jpg')
                #print("Single image path is:", str(img_path))
                name = str(filename).split('/')[len(str(filename).split('/'))-1] 
                #print("name is:", name)                               
                img = np.array(Image.open(img_path).convert("RGB"))
                segs_directory = str(self.segs_dir) +"/"+ str('/'.join(str(filename).split('/')[:-1]) ) + "/" 
                #print("Segs directory:", segs_directory)
                os.makedirs(segs_directory, exist_ok=True)
                segs_t = self.preprocess_data(img, crop_data=True)
                #a = np.uint8(segs_t.numpy()) 
                #Image.fromarray(a).save(f, 'png')
                data = ((segs_t.cpu()).numpy()).reshape(256,256)
                rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
                segs = Image.fromarray(rescaled)
                print(segs)
                #torchvision.utils.save_image (segs, str(self.segs_dir) +"/"+ str(filename) + '.png')
                segs.save(str(self.segs_dir) +"/"+ str(filename) + '.png')
                #imgs_directory = 
              



if __name__ == "__main__":
    ## Parse options ##
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add = parser.add_argument

    Segmentation_Generator.get_args(parser)

    args, _ = parser.parse_known_args()

    ## Initialize the model ##
    generator = Segmentation_Generator(args, 'train')

    ## Perform training ##
    generator.get_segs()
