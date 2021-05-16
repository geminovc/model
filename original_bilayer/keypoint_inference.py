import argparse
import glob
import copy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
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

import os
from datasets import utils as ds_utils
from runners import utils as rn_utils
from external.Graphonomy import wrapper
import face_alignment


args_dict = {
    'project_dir': '.',
    'keypoint_dir': '/data/pantea/video_conf/keypoints/train/',
    'init_networks': 'identity_embedder, texture_generator, keypoints_embedder, inference_generator',
    'init_which_epoch': '2225',
    'num_gpus': 1,
    'which_epoch': '1225',
    'image_size': '256',
    'output_stickmen':False, 
    'enh_apply_masks': False,
    'inf_apply_masks': False}



# Stickman/facemasks drawer
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

#net_seg = wrapper.SegmentationWrapper(args_dict)

to_tensor = transforms.ToTensor()
def preprocess_data(input_imgs, crop_data=True):
        imgs = []
        poses = []
        stickmen = []

        if len(input_imgs.shape) == 3:
            input_imgs = input_imgs[None]
            N = 1

        else:
            N = input_imgs.shape[0]

        for i in range(N):
            pose = fa.get_landmarks(input_imgs[i])[0]

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
                img = Image.fromarray(input_imgs[i])

                if crop_data:
                    img = img.crop((center[0]-size, center[1]-size, center[0]+size, center[1]+size))
                    s = img.size[0]
                    pose -= center - size

                img = img.resize((int(args_dict['image_size']), int(args_dict['image_size'])), Image.BICUBIC)

                imgs.append((to_tensor(img) - 0.5) * 2)

            if crop_data:
                pose = pose / float(s)

            poses.append(torch.from_numpy((pose - 0.5) * 2).view(-1))

        poses = torch.stack(poses, 0)[None]

        if args_dict['output_stickmen']:
            stickmen = ds_utils.draw_stickmen(args_dict, poses[0])
            stickmen = stickmen[None]

        if input_imgs is not None:
            imgs = torch.stack(imgs, 0)[None]

        if args_dict['num_gpus'] > 0:
            poses = poses.cuda()
            
            if input_imgs is not None:
                imgs = imgs.cuda()

                if args_dict['output_stickmen']:
                    stickmen = stickmen.cuda()

        segs = None
        #if hasattr('net_seg') and not isinstance(imgs, list):
        #    segs = net_seg(imgs)[None]

        return poses, imgs, segs, stickmen





images = []
fname = []
<<<<<<< HEAD
for f in glob.iglob("/data/pantea/video_conf/imgs/test/0/0/0/*"):
    images.append(np.asarray(Image.open(f)))
=======
for f in glob.iglob("/video-conf/scratch/pantea/Vedantha_dataset/imgs/train/barack/graduation/0/*"):
    images.append(np.asarray(Image.open(f)))
    print(f)
    fname.append((f.split('/')[len(f.split('/'))-1]).split('.')[0])

images = np.array(images)

for i in range(0, len(images)):
    print(i)
    poses, imgs, segs, stickmen = preprocess_data(images[i], crop_data=True)
    os.makedirs('/video-conf/scratch/pantea/Vedantha_dataset/keypoints/train/barack/graduation/0/', exist_ok=True)
    np.save('/video-conf/scratch/pantea/Vedantha_dataset/keypoints/train/barack/graduation/0/'+fname[i], (poses.cpu()).numpy())



print("barak done!")

images = []
fname = []
for f in glob.iglob("/video-conf/scratch/pantea/Vedantha_dataset/imgs/train/dave/lecture1/0/*"):
    images.append(np.asarray(Image.open(f)))
    print(f)
    fname.append((f.split('/')[len(f.split('/'))-1]).split('.')[0])

images = np.array(images)

for i in range(0, len(images)):
    print(i)
    poses, imgs, segs, stickmen = preprocess_data(images[i], crop_data=True)
    os.makedirs('/video-conf/scratch/pantea/Vedantha_dataset/keypoints/train/dave/lecture1/0/', exist_ok=True)
    np.save('/video-conf/scratch/pantea/Vedantha_dataset/keypoints/train/dave/lecture1/0/'+fname[i], (poses.cpu()).numpy())


print("dave done!")


images = []
fname = []
for f in glob.iglob("/video-conf/scratch/pantea/Vedantha_dataset/imgs/train/gaga/graduation/0/*"):
    images.append(np.asarray(Image.open(f)))
    print(f)
    fname.append((f.split('/')[len(f.split('/'))-1]).split('.')[0])

images = np.array(images)

for i in range(0, len(images)):
    print(i)
    poses, imgs, segs, stickmen = preprocess_data(images[i], crop_data=True)

    os.makedirs('/video-conf/scratch/pantea/Vedantha_dataset/keypoints/train/gaga/graduation/0/', exist_ok=True)
    np.save('/video-conf/scratch/pantea/Vedantha_dataset/keypoints/train/gaga/graduation/0/'+fname[i], (poses.cpu()).numpy())



print("gaga done!")


images = []
fname = []
for f in glob.iglob("/video-conf/scratch/pantea/Vedantha_dataset/imgs/train/kevin/stonks/0/*"):
    images.append(np.asarray(Image.open(f)))
    print(f)
    fname.append((f.split('/')[len(f.split('/'))-1]).split('.')[0])

images = np.array(images)

for i in range(0, len(images)):
    print(i)
    poses, imgs, segs, stickmen = preprocess_data(images[i], crop_data=True)
    os.makedirs('/video-conf/scratch/pantea/Vedantha_dataset//keypoints/train/kevin/stonks/0/', exist_ok=True)
    np.save('/video-conf/scratch/pantea/Vedantha_dataset/keypoints/train/kevin/stonks/0/'+fname[i], (poses.cpu()).numpy())


print("kevin done!")

images = []
fname = []
for f in glob.iglob("/video-conf/scratch/pantea/Vedantha_dataset/imgs/train/sundar/sundarVisualization/0/*"):
    images.append(np.asarray(Image.open(f)))
    print(f)
    fname.append((f.split('/')[len(f.split('/'))-1]).split('.')[0])

images = np.array(images)

for i in range(0, len(images)):
    print(i)
    poses, imgs, segs, stickmen = preprocess_data(images[i], crop_data=True)
    os.makedirs('/video-conf/scratch/pantea/Vedantha_dataset/keypoints/train/sundar/sundarVisualization/0/', exist_ok=True)
    np.save('/video-conf/scratch/pantea/Vedantha_dataset/keypoints/train/sundar/sundarVisualization/0/'+fname[i], (poses.cpu()).numpy())
