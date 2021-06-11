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

print("good")

keypoints_path = '/video-conf/scratch/pantea/temp_per_person_extracts/keypoints/test/id00012/yi_qz725MjE/00163/78.npy'
keypoints = np.load(keypoints_path).astype('float32')
keypoints = keypoints.reshape((68,2)) #I added this
#save_as_image(keypoints, name='good.png')
#save_stickmen(keypoints, name='good_stickmen.png')

nose_top = keypoints[28]
nose_bottom = keypoints[34]
left_side = keypoints[3]
right_side = keypoints[15]

##print("nose_top",nose_top)
#print("nose_bottom",nose_bottom)
#print("right_side",right_side)
#print("left_side",left_side)

nose_vector  = nose_top - nose_bottom
right_vector = right_side - nose_bottom
left_vector  = left_side  - nose_bottom

#print("nose_vector",nose_vector)
#print("right_vector",right_vector)
#print('left_vector',left_vector)

right_angle = -math.atan2(right_vector[1], right_vector[0]) + math.atan2(nose_vector[1], nose_vector[0])
left_angle  = -math.atan2(left_vector[1], left_vector[0])   + math.atan2(nose_vector[1], nose_vector[0])
#print("nose to right ",180/3.14*normalizeAngle(right_angle))
#print("nose to left " ,180/3.14*normalizeAngle(left_angle))

left_width = (keypoints[30,1]-keypoints[2,1])**2 + (keypoints[30,0]-keypoints[2,0])**2
right_width = (keypoints[30,1]-keypoints[14,1])**2 + (keypoints[30,0]-keypoints[14,0])**2

# left-tilted pose
for i in range(1,4):
    if keypoints[30,0]-keypoints[33,0] != 0:
        if left_width < right_width and keypoints[i,1]<(keypoints[30,1]-keypoints[33,1])/(keypoints[30,0]-keypoints[33,0])*(keypoints[i,0]-keypoints[33,0])+keypoints[33,1]:
            if keypoints[i,0] >= min(keypoints[30,0],keypoints[33,0]) and keypoints[i,0] <= max(keypoints[30,0],keypoints[33,0]):
            #if keypoints[i,1] >= min(keypoints[30,1],keypoints[33,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[33,1]):
                print("danger!")
    else:
        if left_width < right_width  and keypoints[i,0] >= keypoints[30,0]:
        #if keypoints[i,1] >= min(keypoints[30,1],keypoints[33,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[33,1]):
            print("danger!")

    if keypoints[30,0]-keypoints[27,0] != 0:
        if left_width < right_width and keypoints[i,1]>(keypoints[30,1]-keypoints[27,1])/(keypoints[30,0]-keypoints[27,0])*(keypoints[i,0]-keypoints[27,0])+keypoints[27,1]:
            if keypoints[i,0] >= min(keypoints[30,0],keypoints[33,0]) and keypoints[i,0] <= max(keypoints[30,0],keypoints[33,0]):
            #if keypoints[i,1] >= min(keypoints[30,1],keypoints[27,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[27,1]):
                print("danger!")
    else:
        if left_width < right_width  and keypoints[i,0] >= keypoints[27,0]:
        #if keypoints[i,1] >= min(keypoints[30,1],keypoints[27,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[27,1]):
            print("danger!")


# # right-tilted pose
# for i in range(12,16):
#     if left_width > right_width and keypoints[i,1]>(keypoints[30,1]-keypoints[33,1])/(keypoints[30,0]-keypoints[33,0])*(keypoints[i,0]-keypoints[33,0])+keypoints[33,1]:
#         print("danger!")
#     if left_width > right_width and keypoints[i,1]<(keypoints[30,1]-keypoints[27,1])/(keypoints[30,0]-keypoints[27,0])*(keypoints[i,0]-keypoints[27,0])+keypoints[27,1]:
#         print("danger!")
    # left_vector  = left_side  - nose_bottom
    # left_angle  = math.atan2(left_vector[1], left_vector[0])   - math.atan2(nose_vector[1], nose_vector[0])
    # #print(left_side)
    # #print("nose to left " ,180/3.14*normalizeAngle(left_angle))

print("bad")
poses = []
keypoints_path = '/video-conf/scratch/pantea/temp_per_person_extracts/keypoints/test/id00012/Z-G8-wqpxwU/00091/0.npy'
keypoints = np.load(keypoints_path).astype('float32')
keypoints = keypoints.reshape((68,2)) #I added this
#save_as_image(keypoints, name='bad.png')
#save_stickmen(keypoints, name='bad_stickmen.png')


nose_top = keypoints[28]
nose_bottom = keypoints[34]
left_side = keypoints[3]
right_side = keypoints[15]

#print("nose_top",nose_top)
#print("nose_bottom",nose_bottom)
#print("right_side",right_side)
#print("left_side",left_side)

nose_vector  = nose_top - nose_bottom
right_vector = right_side - nose_bottom
left_vector  = left_side  - nose_bottom

#print("nose_vector",nose_vector)
#print("right_vector",right_vector)
#print('left_vector',left_vector)

right_angle = -math.atan2(right_vector[1], right_vector[0]) + math.atan2(nose_vector[1], nose_vector[0])
left_angle  = -math.atan2(left_vector[1], left_vector[0])   + math.atan2(nose_vector[1], nose_vector[0])
#print("nose to right ",180/3.14*normalizeAngle(right_angle))
#print("nose to left " ,180/3.14*normalizeAngle(left_angle))


left_width = (keypoints[30,1]-keypoints[2,1])**2 + (keypoints[30,0]-keypoints[2,0])**2
right_width = (keypoints[30,1]-keypoints[14,1])**2 + (keypoints[30,0]-keypoints[14,0])**2

# left-tilted pose
for i in range(1,4):
    if keypoints[30,0]-keypoints[33,0] != 0:
        if left_width < right_width and keypoints[i,1]<(keypoints[30,1]-keypoints[33,1])/(keypoints[30,0]-keypoints[33,0])*(keypoints[i,0]-keypoints[33,0])+keypoints[33,1]:
            if keypoints[i,0] >= min(keypoints[30,0],keypoints[33,0]) and keypoints[i,0] <= max(keypoints[30,0],keypoints[33,0]):
            #if keypoints[i,1] >= min(keypoints[30,1],keypoints[33,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[33,1]):
                print("danger!")
    else:
        if left_width < right_width  and keypoints[i,0] >= keypoints[30,0]:
        #if keypoints[i,1] >= min(keypoints[30,1],keypoints[33,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[33,1]):
            print("danger!")

    if keypoints[30,0]-keypoints[27,0] != 0:
        if left_width < right_width and keypoints[i,1]>(keypoints[30,1]-keypoints[27,1])/(keypoints[30,0]-keypoints[27,0])*(keypoints[i,0]-keypoints[27,0])+keypoints[27,1]:
            if keypoints[i,0] >= min(keypoints[30,0],keypoints[33,0]) and keypoints[i,0] <= max(keypoints[30,0],keypoints[33,0]):
            #if keypoints[i,1] >= min(keypoints[30,1],keypoints[27,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[27,1]):
                print("danger!")
    else:
        if left_width < right_width  and keypoints[i,0] >= keypoints[27,0]:
        #if keypoints[i,1] >= min(keypoints[30,1],keypoints[27,1]) and keypoints[i,1] <= max(keypoints[30,1],keypoints[27,1]):
            print("danger!")
# # right-tilted pose
# for i in range(12,16):
#     if left_width > right_width and keypoints[i,1]>(keypoints[30,1]-keypoints[33,1])/(keypoints[30,0]-keypoints[33,0])*(keypoints[i,0]-keypoints[33,0])+keypoints[33,1]:
#         print("danger!")
#     if left_width > right_width and keypoints[i,1]<(keypoints[30,1]-keypoints[27,1])/(keypoints[30,0]-keypoints[27,0])*(keypoints[i,0]-keypoints[27,0])+keypoints[27,1]:
#         print("danger!")
#     # left_vector  = left_side  - nose_bottom
#     # left_angle  = math.atan2(left_vector[1], left_vector[0])   - math.atan2(nose_vector[1], nose_vector[0])
#     # #print(left_side)
#     # #print("nose to left " ,180/3.14*normalizeAngle(left_angle))