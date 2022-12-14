"""
This function takes the files from the voxceleb structure 
root provided and computes the bins of the yaws split by video. 
This resulting pickle file is passed into voxceleb.py if you are trying to do the rebalancing.

Specifically, once you've run extract_angles.py, you want to pass in that save_root
folder as the root in this file and then the bins will be calculated and then saved
in the result_path + bins.pkl.

The binning is defaulted to bin by yaw.

The command I run is:
python get_bins.py --root /data/vision/billf/video-conf/scratch/vedantha/stabilizing_test_3 --result_path /data/vision/billf/video-conf/scratch/vedantha --spacing 10
"""

from tqdm import tqdm

import numpy as np
import pickle
import argparse
import glob

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Gets bins from the calculated headposes')
    parser.add_argument('--root', dest='root', help='Path of angles', 
        default='/data/vision/billf/video-conf/scratch/vedantha/stabilizing_test_3')
    parser.add_argument('--result_path', dest='result_path', help='Path of angles', 
        default='/data/vision/billf/video-conf/scratch/vedantha')
    parser.add_argument('--spacing', dest='spacing', help='Bin size', 
        type=int, default='10')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    spacing = args.spacing
    all_bins={}
    progress=1
    
    # The odd file structure here is used because it matches the voxceleb.py file structure
    for video in tqdm(glob.glob(args.root + '/angles/*/*/*')):
        for file in glob.glob(video + '/*/*'):
            arr = np.load(file)
            video_uid = (file.split('/'))[-3]

            # Create the bins for that video
            if video_uid not in all_bins.keys():
                all_bins[video_uid] = []
                for _ in range(int(180/spacing)):
                    all_bins[video_uid].append([])

            # Add the frame to its bin
            yaw = arr[0]
            all_bins[video_uid][int(yaw//spacing)].append(file)
            if progress % 1000 == 0:
                print(progress)
            progress += 1
    # Save results
    with open(args.result_path + '/bins.pkl', "wb") as out:
        pickle.dump(all_bins, out)
    # Print bins for sanity check
    for element in all_bins:
        print(element)
        for i in all_bins[element]:
            print(len(i))
