from keypoint_segmentation_generator import keypoint_segmentation_generator

args_dict ={ 
    'video_root': '/video-conf/scratch/test_dataset/mp4/',
    'data_root': '/video-conf/scratch/pantea/video_conf_datasets/general_dataset_test',
    'num_gpus': 1,
    'image_size': 256,
    'sampling_rate': 50,
    'output_segmentation':True,
    'output_stickmen': False,
    'batch_size': 48}


generator = keypoint_segmentation_generator(args_dict, 'test')
generator.get_poses()