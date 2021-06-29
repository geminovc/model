import sys, os, argparse, glob
from torch.multiprocessing import Queue, Process
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import hopenet

from skimage import io
import dlib

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--index', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--proc', help='the number of threads you using',
            default=15, type=int)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='/data/vision/billf/video-conf/scratch/vedantha/hope_weights/hopenet_robust_alpha1.pkl', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
          default='/data/vision/billf/video-conf/scratch/vedantha/hope_weights/mmod_human_face_detector.dat', type=str)
    parser.add_argument('--root', dest='root', help='Path of videos', default='/data/vision/billf/video-conf/scratch/pantea/temp_per_person_1_extracts')
    parser.add_argument('--save_path', dest='save_path', help='Path of angles', default='/data/vision/billf/video-conf/scratch/pantea/temp_per_person_1_extracts')
    args = parser.parse_args()
    return args
def get_angles_from_path(args, idx_tensor, transformations, cnn_face_detector, model):
    frame = cv2.imread(args.img_path)
    cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # Dlib detect
    dets = cnn_face_detector(cv2_frame, 1)
    # Ensure only one face in frame
    if len(dets) != 1:
        return
    for idx, det in enumerate(dets):
        # Get x_min, y_min, x_max, y_max, conf
        x_min = det.rect.left()
        y_min = det.rect.top()
        x_max = det.rect.right()
        y_max = det.rect.bottom()
        conf = det.confidence

        if True: 
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            x_min -= 2 * bbox_width / 4
            x_max += 2 * bbox_width / 4
            y_min -= 3 * bbox_height / 4
            y_max += bbox_height / 4
            x_min = max(x_min, 0); y_min = max(y_min, 0)
            x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)
            # Crop image
            x_min = int(x_min)
            x_max = int(x_max)
            y_min = int(y_min)
            y_max = int(y_max)

            img = cv2_frame[y_min:y_max,x_min:x_max]
            img = Image.fromarray(img)

            # Transform
            img = transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
            img = Variable(img).cuda(gpu)

            yaw, pitch, roll = model(img)

            yaw_predicted = F.softmax(yaw)
            pitch_predicted = F.softmax(pitch)
            roll_predicted = F.softmax(roll)
            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
    return np.asarray([yaw_predicted, pitch_predicted, roll_predicted])

def save_angles(args, idx_tensor, transformations, cnn_face_detector, model):
    img_path  = args.img_path
    split_string = img_path.split('/')
    path_to_angles =['/data/vision/billf/video-conf/scratch/vedantha']+ ['stabilizing_test_2'] + ['angles'] + split_string[-5:-1] + [split_string[-1][:-3] + 'npy']
    path_to_angles_dir = '/'.join(path_to_angles)
    
    if not os.path.exists('/'.join(path_to_angles[:-1])):
        os.makedirs('/'.join(path_to_angles[:-1]))
        print('made dir')    
    angles = get_angles_from_path(args, idx_tensor, transformations, cnn_face_detector, model)
    angles_as_array = np.asarray(angles)
    print(path_to_angles_dir)
    print(('/'.join(path_to_angles[:-1])))
    np.save(path_to_angles_dir, angles_as_array)

    
def do_process(q):
    while True:
        arguments = q.get()
        save_angles(*arguments)
        q.task_done()
        del arguments
if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = 'output/video'
    root = args.root


    if not os.path.exists(root):
        sys.exit('Video does not exist')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Dlib face detection model
    #cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model)

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    print('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    print('Ready to test network.')

    # Test the Model
    total = 0
    mp.set_start_method('spawn')
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    args.img_path = []
    for index, file in enumerate(glob.glob(args.root+'/imgs/*/*/*/*/*')):
        print(file)
        total+=1
        print(total)
        if total % 21 != args.index:
            continue
        args.img_path=file
        
        img_path  = args.img_path
        split_string = img_path.split('/')
        path_to_angles = [args.save_path] + ['angles'] + split_string[-5:-1] + [split_string[-1][:-3] + 'npy']
        path_to_angles_dir = '/'.join(path_to_angles)
        
        if not os.path.exists('/'.join(path_to_angles[:-1])):
            os.makedirs('/'.join(path_to_angles[:-1]), exist_ok=True)
            print('made dir')    
        frame = cv2.imread(args.img_path)
        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        # Dlib detect
        #dets = cnn_face_detector(cv2_frame, 1)
        # Ensure only one face in frame
        #if len(dets) != 1:
        #    continue
        #for idx, det in enumerate(dets):
            # Get x_min, y_min, x_max, y_max, conf
            #x_min = det.rect.left()
            #y_min = det.rect.top()
            #x_max = det.rect.right()
            #y_max = det.rect.bottom()
            #bbox_width = abs(x_max - x_min)
            #bbox_height = abs(y_max - y_min)
            #x_min -= 2 * bbox_width / 4
            #x_max += 2 * bbox_width / 4
            #y_min -= 3 * bbox_height / 4
            #y_max += bbox_height / 4
            #x_min = max(x_min, 0); y_min = max(y_min, 0)
            #x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)
            ## Crop image
            #x_min = int(x_min)
            #x_max = int(x_max)
            #y_min = int(y_min)
            #y_max = int(y_max)
        if True:
            img = cv2_frame#[y_min:y_max,x_min:x_max]
            img = Image.fromarray(img)

            # Transform
            img = transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
            img = Variable(img).cuda(gpu)

            yaw, pitch, roll = model(img)

            yaw_predicted = F.softmax(yaw)
            pitch_predicted = F.softmax(pitch)
            roll_predicted = F.softmax(roll)
            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
        print(yaw_predicted)
        angles = np.asarray([yaw_predicted.cpu().numpy(), pitch_predicted.cpu().numpy(), roll_predicted.cpu().numpy()])
        print(angles)
        print(type(angles[0]))
        np.save(path_to_angles_dir, angles, allow_pickle=True)

