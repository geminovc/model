"""
This file finds the percentage of left-tilted difficult poses in a keypoint directory.
Difficult poses are poses where the cheeks are behind the nose and you can not see a part of them.

To run this file use:

python find_l2_distance.py --keypoint_directory <'KEYPOINTS_DIR'>

Example:
python find_l2_distance.py --keypoint_directory '/video-conf/scratch/pantea/temp_per_person_extracts/keypoints/train'

Inputs
----------
keypoint_directory: path to keypoints

Outputs
----------

It prints the percentage of difficult left-tilted poses.


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
import os

parser= argparse.ArgumentParser("Difficult pose finder")
parser.add_argument('--keypoint_directory',
        default= '/video-conf/scratch/pantea/temp_per_person_extracts/keypoints/train',
        type=str,
        help='keypoints directory')


args = parser.parse_args()

# Function for stickman and facemasks drawing
def draw_stickmen(poses):
    ### Define drawing options ###
    folder_postfix = '2d_crop'
    if not '2d' in folder_postfix and not '3d' in folder_postfix:
        # Arcsoft keypoints
        edges_parts  = [
            merge_parts(range(0, 19), range(103, 121)), # face
            list(range(19, 29)), list(range(29, 39)), # eyebrows
            merge_parts(range(39, 51), range(121, 133)), list(range(165, 181)), [101, 101], # right eye
            merge_parts(range(51, 63), range(133, 145)), list(range(181, 197)), [102, 102], # left eye
            list(range(63, 75)), list(range(97, 101)), # nose
            merge_parts(range(75, 88), range(145, 157)), merge_parts(range(157, 165), range(88, 95))] # lips

        closed_parts = [
            False, 
            True, True, 
            True, True, False, 
            True, True, False, 
            False, False, 
            True, True]

        colors_parts = [
            (  255,  255,  255), 
            (  255,    0,    0), (    0,  255,    0),
            (    0,    0,  255), (    0,    0,  255), (    0,    0,  255),
            (  255,    0,  255), (  255,    0,  255), (  255,    0,  255),
            (    0,  255,  255), (    0,  255,  255),
            (  255,  255,    0), (  255,  255,    0)]

    else:
        edges_parts  = [
            list(range( 0, 17)), # face
            list(range(17, 22)), list(range(22, 27)), # eyebrows (right left)
            list(range(27, 31)) + [30, 33], list(range(31, 36)), # nose
            list(range(36, 42)), list(range(42, 48)), # right eye, left eye
            list(range(48, 60)), list(range(60, 68))] # lips

        closed_parts = [
            False, False, False, False, False, True, True, True, True]

        colors_parts = [
            (  255,  255,  255), 
            (  255,    0,    0), (    0,  255,    0),
            (    0,    0,  255), (    0,    0,  255), 
            (  255,    0,  255), (    0,  255,  255),
            (  255,  255,    0), (  255,  255,    0)]

    ### Start drawing ###
    stickmen = []

    for pose in poses:
        if isinstance(pose, torch.Tensor):
            # Apply conversion to numpy, asssuming the range to be in [-1, 1]
            xy = (pose.view(-1, 2).cpu().numpy() + 1) / 2 * 256
        
        else:
            # Assuming the range to be [0, 1]
            xy = pose[:, :2] * 256

        xy = xy[None, :, None].astype(np.int32)

        stickman = np.ones((256, 256, 3), np.uint8)

        for edges, closed, color in zip(edges_parts, closed_parts, colors_parts):

            stickman = cv2.polylines(stickman, xy[:, edges], closed, color, thickness=2)

        stickman = torch.FloatTensor(stickman.transpose(2, 0, 1)) / 255.
        stickmen.append(stickman)

    stickmen = torch.stack(stickmen)
    stickmen = (stickmen - 0.5) * 2. 

    return stickmen

def normalizeAngle(angle):
    newAngle = angle
    if (newAngle <= -np.pi):
         newAngle += 2*np.pi
    if (newAngle > np.pi):
         newAngle -= 2*np.pi
    return newAngle

def save_as_image(keypoints, name='test.png'):
    image = np.zeros((256,256))
    for i in range(0, len(keypoints)):
        image[int(keypoints[i,0])-1:int(keypoints[i,0])+2,int(keypoints[i,1])-1:int(keypoints[i,1])+2]=1
    image = np.transpose(image)
    #Rescale to 0-255 and convert to uint8
    rescaled = (255.0 / image.max() * (image - image.min())).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save(name)

def save_stickmen(input_keypoints, name='stickmen.png'):
    poses = []
    input_keypoints = input_keypoints[:68, :]
    input_keypoints[:, :2] /= 256
    input_keypoints = input_keypoints[:, :2]
    poses += [torch.from_numpy(input_keypoints.reshape(-1))]
    target_poses = (torch.stack(poses) - 0.5) * 2.0
    stickmen = draw_stickmen(target_poses).cpu().numpy()
    rescaled = (255.0 / stickmen.max() * (stickmen - stickmen.min())).astype(np.uint8)
    im = Image.fromarray(rescaled[0].transpose(1, 2, 0))
    im.save(name)


keypoint_directory=pathlib.Path(args.keypoint_directory)
keypoints_paths = keypoint_directory.glob('*/*/*/*')
keypoints_paths = sorted([str(seq) for seq in keypoints_paths])

number_of_difficult_poses = 0
for keypoint_path in keypoints_paths:
    print(keypoint_path) 
    keypoints = np.load(keypoint_path).astype('float32')
    keypoints = keypoints.reshape((68,2)) 


    left_width = (keypoints[30,1]-keypoints[2,1])**2 + (keypoints[30,0]-keypoints[2,0])**2
    right_width = (keypoints[30,1]-keypoints[14,1])**2 + (keypoints[30,0]-keypoints[14,0])**2
    difficulty = 0
    
    # left-tilted pose
    for i in range(1,4):
        if keypoints[30,0]-keypoints[33,0] != 0:
            if left_width < right_width and keypoints[i,1]<(keypoints[30,1]-keypoints[33,1])/(keypoints[30,0]-keypoints[33,0])*(keypoints[i,0]-keypoints[33,0])+keypoints[33,1]:
                if keypoints[i,0] >= min(keypoints[30,0],keypoints[33,0]) and keypoints[i,0] <= max(keypoints[30,0],keypoints[33,0]):
                #if keypoints[i,1] >= min(keypoints[30,1],keypoints[33,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[33,1]):
                    print("danger")
                    difficulty+=1
        else:
            if left_width < right_width  and keypoints[i,0] >= keypoints[30,0]:
            #if keypoints[i,1] >= min(keypoints[30,1],keypoints[33,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[33,1]):
                print("danger!")
                difficulty+=1

        if keypoints[30,0]-keypoints[27,0] != 0:
            if left_width < right_width and keypoints[i,1]>(keypoints[30,1]-keypoints[27,1])/(keypoints[30,0]-keypoints[27,0])*(keypoints[i,0]-keypoints[27,0])+keypoints[27,1]:
                if keypoints[i,0] >= min(keypoints[30,0],keypoints[33,0]) and keypoints[i,0] <= max(keypoints[30,0],keypoints[33,0]):
                #if keypoints[i,1] >= min(keypoints[30,1],keypoints[27,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[27,1]):
                    print("danger!")
                    difficulty+=1
        else:
            if left_width < right_width  and keypoints[i,0] >= keypoints[27,0]:
            #if keypoints[i,1] >= min(keypoints[30,1],keypoints[27,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[27,1]):
                print("danger!")
                difficulty+=1

    if difficulty >0:
        number_of_difficult_poses+=1

print("percentage of difficult poses: ", str(number_of_difficult_poses/len(keypoints_paths)))