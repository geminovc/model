import numpy as np
import torch
import os
import scipy.io as sio
import cv2
import math
from math import cos, sin
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from importlib import import_module
import cv2



def get_dataloader(args, phase):
    dataset = import_module(f'datasets.{args.dataloader_name}').DatasetWrapper(args, phase)
    batch_size = args.batch_size // args.world_size
    if phase == 'metrics':
        # Sets batch size to 1 if you are passing metrics data because we want to view all the images separately
        batch_size = 1
    if phase == 'train': 
    	args.train_size = len(dataset)
    return DataLoader(dataset, 
        batch_size=batch_size, 
        sampler=DistributedSampler(dataset, args.world_size, args.rank, shuffle=False), # shuffling is done inside the dataset
        num_workers=args.num_workers_per_process,
        drop_last=False)

# Required to draw a stickman for ArcSoft keypoints
def merge_parts(part_even, part_odd):
    output = []
    
    for i in range(len(part_even) + len(part_odd)):
        if i % 2:
            output.append(part_odd[i // 2])
        else:
            output.append(part_even[i // 2])

    return output

# Function for stickman and facemasks drawing
def draw_stickmen(args, poses):
    ### Define drawing options ###
    if not '2d' in args.folder_postfix and not '3d' in args.folder_postfix:
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
            xy = (pose.view(-1, 2).cpu().numpy() + 1) / 2 * args.image_size
        
        else:
            # Assuming the range to be [0, 1]
            xy = pose[:, :2] * self.args.image_size

        xy = xy[None, :, None].astype(np.int32)

        stickman = np.ones((args.image_size, args.image_size, 3), np.uint8)

        for edges, closed, color in zip(edges_parts, closed_parts, colors_parts):
            stickman = cv2.polylines(stickman, xy[:, edges], closed, color, thickness=args.stickmen_thickness)

        stickman = torch.FloatTensor(stickman.transpose(2, 0, 1)) / 255.
        stickmen.append(stickman)

    stickmen = torch.stack(stickmen)
    stickmen = (stickmen - 0.5) * 2. 

    return stickmen

# Flip vector poses via x axis
def flip_poses(args, keypoints, size):
    if not '2d' in args.folder_postfix and not '3d' in args.folder_postfix:
        # Arcsoft keypoints
        edges_parts  = [
            merge_parts(range(0, 19), range(103, 121)), # face
            list(range(19, 29)), list(range(29, 39)), # eyebrows
            merge_parts(range(39, 51), range(121, 133)), list(range(165, 181)), [101, 101], # right eye
            merge_parts(range(51, 63), range(133, 145)), list(range(181, 197)), [102, 102], # left eye
            list(range(63, 75)), list(range(97, 101)), # nose
            merge_parts(range(75, 88), range(145, 157)), merge_parts(range(157, 165), range(88, 95))] # lip

    else:
        edges_parts  = [
            list(range( 0, 17)), # face
            list(range(17, 22)), list(range(22, 27)), # eyebrows (right left)
            list(range(27, 31)) + [30, 33], list(range(31, 36)), # nose
            list(range(36, 42)), list(range(42, 48)), # right eye, left eye
            list(range(48, 60)), list(range(60, 68))] # lips


    keypoints[:, 0] = size - keypoints[:, 0]

    # Swap left and right face parts
    if not '2d' in args.folder_postfix and not '3d' in args.folder_postfix:
        l_parts  = edges_parts[1] + edges_parts[3] + edges_parts[4] + edges_parts[5][:1]
        r_parts = edges_parts[2] + edges_parts[6] + edges_parts[7] + edges_parts[8][:1]

    else:
        l_parts = edges_parts[2] + edges_parts[6]
        r_parts = edges_parts[1] + edges_parts[5]

    keypoints[l_parts + r_parts] = keypoints[r_parts + l_parts]

    return keypoints
def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result

def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

def mse_loss(input, target):
    return torch.sum(torch.abs(input.data - target.data) ** 2)

def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size
    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img
