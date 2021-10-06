import glob
import pathlib
import numpy as np
import pickle
import pdb
import os
import argparse
import shutil 
 
parser= argparse.ArgumentParser("Make new per_video dataset")
parser.add_argument('--data_root',
        default= '/data/pantea/per_video_2_three_datasets',
        type=str,
        help=' root to imgs/keypoints/segs')

parser.add_argument('--yaw_root',
        default= '/data/pantea/per_video_2_three_datasets_yaws/angles',
        type=str,
        help='root to yaw numpy values')

parser.add_argument('--target_video_id',
        default= 'id00015/M_u0SV9wLro',  
        type=str,
        help='target video id with the person id')

parser.add_argument('--num_test_sessions',
        default= 2,
        type=int,
        help='number of sessions to be held out for test')

args = parser.parse_args()

def load_pickle(path_string):
    pkl_file = open(path_string, 'rb')
    my_dict = pickle.load(pkl_file)
    pkl_file.close()
    return my_dict

def move_directories(session_ids, source_phase, destination_phase):
    for session_id in session_ids:
        # move imgs, keypoints, and segs
        for x in ['imgs', 'keypoints', 'segs']:
            source = pathlib.Path(args.data_root + '/' + x + '/' + source_phase + '/' + session_id)
            destination = pathlib.Path(args.data_root + '/' + x + '/' + destination_phase + '/' + session_id)
            print ("moving {} to {}".format(source , destination))
            try:
                shutil.move(str(source), str(destination))
            except:
                pass 
        
        # move yaws
        source = pathlib.Path(args.yaw_root + '/' + source_phase + '/' + session_id)
        destination = pathlib.Path(args.yaw_root + '/' + destination_phase + '/' + session_id)
        try:
            shutil.move(str(source), str(destination))
        except:
            pass
        print ("moving {} to {}".format(source , destination))

# Move train to unseen 
train_session_ids =  pathlib.Path(args.data_root + '/imgs/train').glob('*/*/*')
train_session_ids = sorted(['/'.join(str(seq).split('/')[-3:]) for seq in train_session_ids])
print(train_session_ids) 
move_directories(train_session_ids, 'train', 'unseen_test')

# Now delete everything in the train
train_session_ids =  pathlib.Path(args.data_root + '/imgs/train').glob('*/*')
train_session_ids = sorted(['/'.join(str(seq).split('/')[-2:]) for seq in train_session_ids])
print("remove from train", train_session_ids) 
for video in train_session_ids:
    try:
        shutil.rmtree(str(args.yaw_root + '/' + 'train' + '/' + video))
    except:
        pass
    for x in ['imgs', 'keypoints', 'segs']:
        try:
            shutil.rmtree(str(args.data_root + '/' + x + '/' + 'train' + '/' + video))
        except:
            pass
        print("removing", str(args.data_root + '/' + x + '/' + 'train' + '/' + video))


# Move test to unseen
test_session_ids =  pathlib.Path(args.data_root + '/imgs/test').glob('*/*/*')
test_session_ids = sorted(['/'.join(str(seq).split('/')[-3:]) for seq in test_session_ids])
move_directories(test_session_ids, 'test', 'unseen_test')

# Now delete everything in the test
test_session_ids =  pathlib.Path(args.data_root + '/imgs/test').glob('*/*')
test_session_ids = sorted(['/'.join(str(seq).split('/')[-2:]) for seq in test_session_ids])
print("remove from test", test_session_ids) 

for video in test_session_ids:
    try:
        shutil.rmtree(str(args.yaw_root + '/' + 'test' + '/' + video))
    except:
        pass
    for x in ['imgs', 'keypoints', 'segs']:
        try:
            shutil.rmtree(str(args.data_root + '/' + x + '/' + 'test' + '/' + video))
        except:
            pass
        print("removing", str(args.data_root + '/' + x + '/' + 'test' + '/' + video))


# Move an unseen_test video to test and train
new_train_session_ids =  pathlib.Path(args.data_root + '/imgs/unseen_test/' + str(args.target_video_id)).glob('*')
new_train_session_ids = sorted(['/'.join(str(seq).split('/')[-3:]) for seq in new_train_session_ids])
train_session_ids = new_train_session_ids[:-args.num_test_sessions]
print("new train_session_ids", train_session_ids)
test_session_ids = new_train_session_ids[-args.num_test_sessions:]
print("new test_session_ids", test_session_ids)
move_directories(train_session_ids, 'unseen_test' , 'train')
move_directories(test_session_ids, 'unseen_test' , 'test')

try:
    shutil.rmtree(str(args.yaw_root + '/' + 'unseen_test' + '/' + str(args.target_video_id)))
except:
    pass
for x in ['imgs', 'keypoints', 'segs']:
    try:
        shutil.rmtree(str(args.data_root + '/' + x + '/' + 'unseen_test' + '/' + str(args.target_video_id)))
    except:
        pass
    print("removing", str(args.data_root + '/' + x + '/' + 'unseen_test' + '/' + str(args.target_video_id)))
