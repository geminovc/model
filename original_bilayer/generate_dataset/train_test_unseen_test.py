'''
This scripts seperates the train datset in data_root in the following format:
    
    1) unseen test videos:(100*unseen_test_train_ratio) percent of total train data is held out for test in data_root/[imgs, keypints, segs]/unseen_test
    2) from the rest of the videos, one random session is collected for test and stored in data_root/[imgs, keypints, segs]/test

How to run this script?
python train_test_unseen_test.py --data_root <PATH_TO_DATA> --random_seed <YOUR_RANDOM_SEED> --unseen_test_train_ratio <UNSEEN_RATIO>

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
        default="/data/pantea/per_person_1_three_datasets_2",
        help='Data root directory')
parser.add_argument('--random_seed',
        type=int,
        default=0,
        help='The random seed')
parser.add_argument('--unseen_test_train_ratio',
        type=float,
        default=0.18,
        help='The ratio of vidos that are seprated as unseen videos')


args = parser.parse_args()
data_root = args.data_root
random_seed = args.random_seed
unseen_test_train_ratio = args.unseen_test_train_ratio


# Set random seed
random.seed(random_seed)

# Data paths
# Images 
imgs_dir = data_root + '/imgs'  
imgs_dir_train = data_root + '/imgs/train/' 
imgs_dir_test = data_root + '/imgs/test/' 
imgs_dir_unseen_test = data_root + '/imgs/unseen_test/' 

# Keypoints
poses_dir = data_root + '/keypoints'  
poses_dir_train = data_root + '/keypoints/train/' 
poses_dir_test = data_root + '/keypoints/test/'
poses_dir_unseen_test = data_root + '/keypoints/unseen_test/'

# Segmentations
segs_dir = data_root + '/segs'  
segs_dir_train = data_root + '/segs/train/' 
segs_dir_test = data_root + '/segs/test/' 
segs_dir_unseen_test = data_root + '/segs/unseen_test/'

# Move some of the videos entirely for unseen_test
all_videos = pathlib.Path(imgs_dir_train).glob('*/*')
all_videos = ['/'.join(str(seq).split('/')[-2:]) for seq in all_videos]
all_videos = sorted(all_videos)
unseen_test_videos = random.sample(all_videos, int(unseen_test_train_ratio*len(all_videos)))

for folder in unseen_test_videos:
    print(folder)
    try:
        # images
        source = imgs_dir_train + folder
        destination = imgs_dir_unseen_test + folder
        print("moving ",source," to " ,destination)
        dest = shutil.move(source, destination) 

        # keypoints
        source = poses_dir_train + folder
        destination = poses_dir_unseen_test + folder
        print("moving ",source," to " ,destination)
        dest = shutil.move(source, destination) 

        # segs
        source = segs_dir_train + folder
        destination = segs_dir_unseen_test + folder
        print("moving ",source," to " ,destination)
        dest = shutil.move(source, destination) 
    except:
        print("Something happend :)")


# Remanining train/test videos 

train_videos = [item for item in all_videos if item not in unseen_test_videos]

for video in train_videos:
    print(video)
    sessions = pathlib.Path(imgs_dir_train + "/" + video).glob('*')
    sessions = ['/'.join(str(seq).split('/')[-3:]) for seq in sessions]
    sessions = sorted(sessions)
    if len(sessions)!=1:
        test_session = random.sample(sessions, 1)[0]
        try:
            # images
            source = imgs_dir_train + test_session
            destination = imgs_dir_test + test_session
            print("moving ",source," to " ,destination)
            dest = shutil.move(source, destination) 

            # keypoints
            source = poses_dir_train + test_session
            destination = poses_dir_test + test_session
            print("moving ",source," to " ,destination)
            dest = shutil.move(source, destination) 

            # segs
            source = segs_dir_train + test_session
            destination = segs_dir_test + test_session
            print("moving ",source," to " ,destination)
            dest = shutil.move(source, destination) 
        except:
            print("Something happend :)")




