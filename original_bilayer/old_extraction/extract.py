from keypoint_segmentation_generator import Keypoint_Segmentation_Generator

args_dict ={ 
    'video_root': '/video-conf/vedantha/voxceleb2/dev/mp4/',
    'data_root': '/data/pantea/video_conf',
    'imgs_dir': '/data/pantea/video_conf',
    'keypoint_dir': 'keypoints',
    'segmentatio_dir':'segs',
    'num_gpus': 1,
    'image_size': 256,
    'sampling_rate': 50,
    'output_segmentation':True,
    'output_stickmen': False,
    'batch_size': 48}


generator = Keypoint_Segmentation_Generator(args_dict, 'train')
generator.get_poses()