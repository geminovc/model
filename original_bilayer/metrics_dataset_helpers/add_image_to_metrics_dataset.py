import os
import sys

# This file allows you to just enter in the paths of the image and the metrics dataset, and then automatically fetches
# the keypoints and segmentations to save in their respective positions
# The path_to_image is the path to the image (ideally enter the full path)
# The position in metrics is just the name of the sequence (i.e. general_1)
# The metrics dir is the root metrics directory

if __name__ == '__main__':
    path_to_image = sys.argv[1]
    position_in_metrics = sys.argv[2]
    metrics_dir = sys.argv[3]
    split_string = path_to_image.split('/')
    path_to_keypoints= split_string[:-6] + ['keypoints'] + split_string[-5:-1] + [split_string[-1][:-3] + 'npy']

    path_to_segs= split_string[:-6] + ['segs'] + split_string[-5:-1] + [split_string[-1][:-3] + 'png']

    # Now that we have the paths we have to save them in the metrics
    keypoint_dir = metrics_dir + '/keypoints/metrics/' + position_in_metrics + '/clip/frame/' + path_to_keypoints[-1]
    path_to_keypoints = '/'.join(path_to_keypoints)
    segs_dir = metrics_dir + '/segs/metrics/' + position_in_metrics + '/clip/frame/' + path_to_segs[-1]
    path_to_segs = '/'.join(path_to_segs)
    imgs_dir = metrics_dir + '/imgs/metrics/' + position_in_metrics + '/clip/frame/' + split_string[-1]
    os.system("cp " + path_to_keypoints + " " + keypoint_dir)
    os.system("cp " + path_to_segs + " " + segs_dir)
    os.system("cp " + path_to_image + " " + imgs_dir)
