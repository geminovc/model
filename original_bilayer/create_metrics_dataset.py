import os
import sys

if __name__ == '__main__':
    path_to_image = sys.argv[1]
    position_in_metrics = sys.argv[2]
    metrics_dir = sys.argv[3]
    split_string = path_to_image.split('/')
    ITEM_INDEX = -6
    
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
