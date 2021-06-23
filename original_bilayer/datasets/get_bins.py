import numpy as np
import pickle

import glob

if __name__ == '__main__':
    spacing = 10
    all_bins=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    i=1
    for file in glob.glob('/data/vision/billf/video-conf/scratch/vedantha/stabilizing_test_3/angles/*/*/*/*/*'):
        arr = np.load(file)
        all_bins[int(arr[1]//10)].append(file)
        if i % 1000 == 0:
            print(i)
        i += 1
    with open('../../bins.pkl', "wb") as out:
        pickle.dump(all_bins, out)
