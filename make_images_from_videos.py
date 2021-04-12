import face_alignment
import cv2
import os
import numpy as np
from PIL import Image
import sys

video_file = sys.argv[1]
save_directory = sys.argv[2]

if __name__ == '__main__':
    """
    Opens up the video, and stores the first n frames which have sundar pichai's face (adding in an offset because the first few frames are edited weirdly)
    """
    video = cv2.VideoCapture(video_file)
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device = 'cuda:2')
    os.makedirs(save_directory, exist_ok=True)
    # Initial parameters: Totals is the number of images you want to keep. Face_offset is the number of images to skip at the beginning (because of weird editing)
    index = 0
    fi = 0
    offset = 0 if len(sys.argv) < 4 else int(sys.argv[3])
    while video.isOpened():
        ret, frame = video.read()
        if frame is None:
            break
        if offset > 0:
            offset-= 1
            continue
        if index % 30 == 0:
            print(index)
        
        pose = fa.get_landmarks(frame)
        if pose is not None and len(pose) == 1:
            
            # Switches frame to RGB from BGR because CV2 reads in BGR
            frame = frame [:,:,::-1]
            img = Image.fromarray(frame)
            img.save(save_directory + str(index) + '.jpg')

            index+=1
    video.release()

    