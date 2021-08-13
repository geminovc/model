'''
This scripts seperates the train datset in data_root. It randomly selects some images as test images and moves them into test folder
    


How to run this script?
python train_test_unseen_test.py --data_root <PATH_TO_DATA> --test_train_ratio <TEST_RATIO>

Note that this script only moves the images in data_root/[imgs, keypints, segs]/train.
'''
import os
import shutil
import  pathlib
import random
import argparse

# Parser options
parser= argparse.ArgumentParser("Splitter")
parser.add_argument('--data_root',
        type=str,
        default="/video-conf/scratch/pantea/random_sampled_per_person_2",
        help='Data root directory')
parser.add_argument('--test_train_ratio',
        type=float,
        default=0.05,
        help='The ratio of total frames that are seprated as test')

args = parser.parse_args()
data_root = args.data_root
test_train_ratio = args.test_train_ratio

# Data paths
imgs_dir = data_root + '/imgs'  
imgs_dir_train = data_root + '/imgs/train/' 
imgs_dir_test = data_root + '/imgs/test/' 

poses_dir = data_root + '/keypoints'  
poses_dir_train = data_root + '/keypoints/train/' 
poses_dir_test = data_root + '/keypoints/test/'

segs_dir = data_root + '/segs'  
segs_dir_train = data_root + '/segs/train/' 
segs_dir_test = data_root + '/segs/test/' 

# List of all the train frames
frames_sequences = pathlib.Path(imgs_dir_train).glob('*/*/*/*')
frames_sequences = [('/'.join(str(seq).split('/')[-4:])).split('.')[0] for seq in frames_sequences]
frames_sequences = sorted(frames_sequences)

if len(frames_sequences) ==0:
    print("No frames in the dataset. Are you sure about the data-root?")

# Sample test frames
test_frames_sequences = random.sample(frames_sequences, int(test_train_ratio*len(frames_sequences)))


print("Moving "+ str(test_train_ratio*100) +" percent of train images to test.")

for frame in test_frames_sequences:

    try:
        # images
        source = imgs_dir_train + frame + '.jpg'
        destination = imgs_dir_test + frame + '.jpg'
        os.makedirs(('/'.join(str(imgs_dir_test + frame ).split('/')[:-1])), exist_ok=True)
        print("moving ",source," to " ,destination)
        dest = shutil.move(source, destination) 

        # keypoints
        source = poses_dir_train + frame + '.npy'
        destination = poses_dir_test + frame + '.npy'
        os.makedirs(('/'.join(str(poses_dir_test + frame ).split('/')[:-1])), exist_ok=True)
        print("moving ",source," to " ,destination)
        dest = shutil.move(source, destination) 

        # segs
        source = segs_dir_train + frame + '.png'
        destination = segs_dir_test + frame + '.png'
        os.makedirs(('/'.join(str(segs_dir_test + frame ).split('/')[:-1])), exist_ok=True)
        print("moving ",source," to " ,destination)
        dest = shutil.move(source, destination) 
    except:
        print("Something happend :)")

