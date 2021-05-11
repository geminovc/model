import face_alignment as fa
import cv2
import os
import numpy as np
from PIL import Image
import sys

video_file = sys.argv[1]
imgs_directory = sys.argv[2]
keypoints_directory = sys.argv[3]
segs_directory = sys.argv[4] 
image_size = int(sys.argv[5])
output_stickmen = False #bool(sys.argv[6])
output_segmentation = False #bool(sys.argv[6])


def preprocess_data(input_imgs, crop_data=True):
        imgs = []
        poses = []
        stickmen = []
        
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device = 'cuda:2')
        pose = fa.get_landmarks(input_imgs)[0]
        to_tensor = transforms.ToTensor()

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
            img = Image.fromarray(input_imgs)

            if crop_data:
                img = img.crop((center[0]-size, center[1]-size, center[0]+size, center[1]+size))
                s = img.size[0]
                pose -= center - size

            img = img.resize((image_size ,image_size), Image.BICUBIC)

            imgs.append((to_tensor(img) - 0.5) * 2)

        if crop_data:
            pose = pose / float(s)

        poses.append(np.reshape(((pose - 0.5) * 2),-1))

        if output_stickmen:
            stickmen = ds_utils.draw_stickmen(args, poses[0])
            stickmen = stickmen[None]

        if input_imgs is not None:
            imgs.append(input_imgs)


        segs = None
        if output_segmentation:
            segs = [] #
        return poses, imgs, segs, stickmen

def  make ():  
    video = cv2.VideoCapture(video_file)
    os.makedirs(imgs_directory, exist_ok=True)
    os.makedirs(keypoints_directory, exist_ok=True)
    os.makedirs(segs_directory, exist_ok=True)
    # Initial parameters: Totals is the number of images you want to keep. Face_offset is the number of images to skip at the beginning (because of weird editing)
    index = 0
    fi = 0
    offset = 0 if len(sys.argv) < 7 else int(sys.argv[7])
    while video.isOpened():
        ret, frame = video.read()
        if frame is None:
            break
        if offset > 0:
            offset-= 1
            continue
        if index % 30 == 0:
            print(index)
        
        frame = frame [:,:,::-1]
        frame = Image.fromarray(frame)
        poses, imgs, segs, stickmen = preprocess_data(frame, crop_data=True)
        if pose is not None and len(pose) == 1:
            
            # Switches frame to RGB from BGR because CV2 reads in BGR
            imgs.save(imgs_directory + str(index) + '.jpg')
            np.save(keypoint_dir + str(index), (poses.cpu()).numpy())
            if output_segmentation:
                segs.save(segs_directory + str(index) + '.png')

            index+=1
    video.release()


make()