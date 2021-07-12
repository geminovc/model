"""
This file gets the yaws, pitch and rolls for all the images in the input directory.

You need to pass in the location of the hopenet model which I've stored at 

/video-conf/scratch/vedantha/hope_weights/hopenet_robust_alpha1.pkl

Specifically, this script takes the images from the voxceleb dataroot provided and computes the
angles, creating a mirror of the voxceleb structure with the same folder names and everything, but
instead of the "imgs" folder you have "angles" and the actual files contain a numpy array with the yaw, pitch
and roll respectively.

Then you want to pass in the root of that newly created vox celeb structured angles into get_bins.py, which
you probably want to modify depending on what you're trying to bin.

You probably shouldn't run this script directly, only through run_all.sh, where I've provided the command I run
"""

from tqdm import tqdm
import sys, os, argparse, glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
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
    parser.add_argument('--data_root', dest='data_root', help='Path of videos', default='/data/vision/billf/video-conf/scratch/pantea/temp_per_person_1_extracts')
    parser.add_argument('--save_root', dest='save_root', help='Path of saved angels root', default='/data/vision/billf/video-conf/scratch/vedantha/stabilizing_test_3')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = 'output/video'
    root = args.root

    if not os.path.exists(root):
        sys.exit('root does not exist')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()

    print('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    print('Ready to run.')
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    # Run the Model
    total = 0
    args.img_path = []
    for index, file in tqdm(enumerate(glob.glob(args.data_root+'/imgs/*/*/*/*/*'))):
        total+=1
        if total % args.proc != args.index:
            continue
        args.img_path=file
        
        img_path  = args.img_path
        split_string = img_path.split('/')
        path_to_angles = [args.save_root] + ['angles'] + split_string[-5:-1] + [split_string[-1][:-3] + 'npy']
        path_to_angles_dir = '/'.join(path_to_angles)
        
        if not os.path.exists('/'.join(path_to_angles[:-1])):
            os.makedirs('/'.join(path_to_angles[:-1]), exist_ok=True)
        frame = cv2.imread(args.img_path)
        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        img = cv2_frame
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
        angles = np.asarray([yaw_predicted.cpu().numpy(), pitch_predicted.cpu().numpy(), roll_predicted.cpu().numpy()])
        print(angles)
        np.save(path_to_angles_dir, angles, allow_pickle=True)

