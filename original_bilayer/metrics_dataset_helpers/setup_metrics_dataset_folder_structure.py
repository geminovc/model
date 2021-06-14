import os
import sys
# Sets up the vox celeb folder structure for metrics
# Pass in the root directory as the first argument and then it should create
# root -> imgs/keypoints/segs -> metrics -> general/per_person/... -> clip -> frame
if __name__ == '__main__':
    metrics_dir = sys.argv[1]
    types = ['imgs', 'keypoints', 'segs']
    
    options = ['general', 'per_person', 'per_video']
    for type in types:
        for item in options:
            for index in range(1,4):
                main_dir = metrics_dir + '/' + type + '/metrics' + \
                    '/' + item + '_' + str(index) + '/clip/frame'
                os.makedirs(main_dir, exist_ok=True)
