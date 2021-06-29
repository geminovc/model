# Running the bilayer model

## Setup
### Conda Environment
Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and 
initialize your shell by running `conda init <YOUR_SHELL_NAME>`.

Clone and setup new environment, check that it has been created and activate it.
```bash
conda env create -f environment.yml
conda env list
conda activate bilayer
```
If you are using the system's anaconda and you don't have write permission, you can create the conda in another directory using:

```
conda env create --prefix /tmp/test-env  -f environment.yml
```
If you can not create an environment because of root privilages, you can use one of the existing environments. On mapmaker use:

```
conda activate torch
```
### Pre-trained Weights 
Download the `pretrained-weights` folder from [here](https://drive.google.com/drive/folders/11SwIYnk3KY61d8qa17Nlb0BN9j57B3L6). You should put the name of this downloaded directory in `--pretrained_weights_dir` in the scripts. 
The pre-trained weights are courtesy [Fast Bi-layer Neural Synthesis of One-Shot Realistic Head Avatars](https://arxiv.org/abs/2008.10174) by Zakharov et. al.

If you want to train your model based on the paper's released checkpoints, you should download the `runs` folder from [here](https://drive.google.com/drive/folders/11SwIYnk3KY61d8qa17Nlb0BN9j57B3L6), and put the folder in the same directory as your downloaded `pretrained-weights` above.  

### Datasets

The model expects the frames to be located in a directory organization similar to [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html). We have created three datasets matching the VoxCeleb2 format. The structure is as the following:

`DATA_ROOT/[imgs, keypoints, segs]/[train, test]/PERSON_ID/VIDEO_ID/SEQUENCE_ID/FRAME_NUM[.jpg, .npy, .png]`

`DATA_ROOT` is the root of your dataset. We have three datasets so far for different experiments: 
* General Dataset: This dataset contains all the data in the VoxCeleb2. We sampled the frames with a `--sampling_rate`. It is currently stored in `/video-conf/scratch/pantea/video_conf_datasets/general_dataset`.
* Per-person Dataset: This dataset contains all the data from one person. We usually use this dataset for our personalization. It is currently stored in `/video-conf/scratch/pantea/video_conf_datasets/per_person_dataset`.
* Per-video Dataset: This dataset contains all the data from one video. We usually use this dataset for online learning. It is currently stored in `/video-conf/scratch/pantea/video_conf_datasets/per_video_dataset`.

In each of the `DATA_ROOT`s above, there are three folders `[imgs, keypoints, segs]` contating the `keypoints` as `.png` and `seg`mentations as `.npy` corresponding to the `imgs` as `.jpg`. In each of them, there are `[train, test]` data in seperate folders in the format of VoxCeleb2 `PERSON_ID/VIDEO_ID/SEQUENCE_ID`.  

### Making your datasets from videos

If you want to make a dataset from videos to use as your train or test data, your videos should be stored in this format:

`VIDEO_ROOT/PERSON_ID/VIDEO_ID/SEQUENCE_ID[.mp4]`

An example of the video structure is:

```
VIDEO_ROOT _ id00012 _ abc _ 00001.mp4
           |         |
           |         |_ def  _ 00001.mp4
           |                |_ 00002.mp4
           |               
           |_ id00013 _ lmn _ 00001.mp4
           |          |
           |          |_ opq  _ 00001.mp4
           |                 |_ 00002.mp4
           |                 |_ 00003.mp4
           |
           |_ id00014 _ rst _ 00001.mp4
                      |    |_ 00002.mp4
                      |
                      |_ uvw  _ 00001.mp4
                             |_ 00002.mp4
                             |_ 00003.mp4
 ```
After formating your videos in such order, you can generate the `[imgs, keypoints, segs]` using our `keypoints_segmentations_generator` module. Before running the bash file `extract.sh` in `keypoints_segmentations_generator` folder, there are a few arguments you need to set in `extract.sh` such as the following:

```
    --pretrained_weights_dir <PATH_TO_YOUR_PRETRAINED_WEIGHTS> \
    --video_root <PATH_TO_YOUR_VIDEO_ROOT> \
    --sampling_rate <YOUR_SAMPLING_RATE> \
    --phase <'train' or 'test> \
    --batch_size <YOUR_BATCH_SIZE> \
    --data_root <PATH_TO_WHERE_YOU_WANT_TO_SAVE_DATASET>  \
    --output_segmentation True \
    --num_gpus <YOUR_NUM_GPUS>
```


Please make sure to set `video_root` the same as the `VIDEO_ROOT` where your videos are stored. If you want to save the videos as train/test dataset, set `phase` to `'train'` or `'test'`. 

The final dataset structure would be similar to:

```
 DATA_ROOT - [imgs, keypoints, segs] _ phase _ id00012 _ abc _ 00001 _ 0 [.jpg, .npy, .png]
                                            |         |            |_ 1 [.jpg, .npy, .png]
                                            |         |            |_ ...
                                            |         |            |_ 99 [.jpg, .npy, .png]
                                            |         |
                                            |         |_ def  _ 00001 _ 0 [.jpg, .npy, .png]
                                            |                |       |_ 1 [.jpg, .npy, .png]
                                            |                |       |_ ...
                                            |                |       |_ 150 [.jpg, .npy, .png]
                                            |                |
                                            |                |_ 00002 _ 0 [.jpg, .npy, .png]
                                            |                        |_ 1 [.jpg, .npy, .png]
                                            |                        |_ ... 
                                            |                        |_ 89 [.jpg, .npy, .png]
                                            |               
                                            |_ id00013 _ lmn _ 00001 _ 0 [.jpg, .npy, .png]
                                            |          |             |_ 1 [.jpg, .npy, .png]
                                            |          |             |_ ... 
                                            |          |             |_ 89 [.jpg, .npy, .png]
                                            |          |
                                            |          |_ opq  _ 00001 _ ...
                                            |                 |_ 00002 _ ...
                                            |                 |_ 00003 _ ...
                                            |
                                            |_ id00014 _ rst _ 00001 _ ...
                                                        |    |_ 00002 _ ...
                                                        |
                                                        |_ uvw  _ 00001 _ 0 [.jpg, .npy, .png]
                                                                |       |_ 1 [.jpg, .npy, .png]
                                                                |       |_ ... 
                                                                |       |_ 68 [.jpg, .npy, .png]
                                                                |
                                                                |_ 00002 _ 0 [.jpg, .npy, .png]
                                                                |       |_ ...
                                                                |       |_ 299 [.jpg, .npy, .png]
                                                                |
                                                                |_ 00003 _ 0 [.jpg, .npy, .png]
                                                                        |_ ...
                                                                        |_ 100 [.jpg, .npy, .png]
 ```

The current version uses user inputs and has fixed `pretrained_weights_dir`. You can simply run the following commnad to extract the keypoints and segmentations:
```
./extract.sh phase <PATH_TO_YOUR_VIDEO_ROOT> <YOUR_SAMPLING_RATE> <PATH_TO_WHERE_YOU_WANT_TO_SAVE_DATASET>

```

Example:
```
./extract.sh 'train' /video-conf/scratch/voxceleb2_test_videos/mp4 50 /video-conf/scratch/pantea/temp_general_extracts_test_2

```


## Training 
If you want to train your model from scratch, you should run:
```bash
cd scripts
CUDA_VISIBLE_DEVICES= <YOUR_CUDA_ID>  bash train_base_model_8gpus.sh 
```
If you want to train your model from paper's released checkpoints, you should run:
```bash
cd scripts
CUDA_VISIBLE_DEVICES=<YOUR_CUDA_ID> bash train_with_pretrained_weights_of_paper.sh
```

### Flags in script files 

We introduced some sets of flags for training:
* `experiment_name`: name of your experiment, we suggest you change your name to something meaningful to distinguish between your experiments
* `pretrained_weights_dir`: After downloading the [pre-trained weights](### Pre-trained Weights), you should edit this directory to point to the correct directory containing the pretrained weights.
* `images_log_rate`: It is the rate that train images are saved in `metrics` tensorboard folder. 
* `metrics_log_rate`: It is the rate that model metrics such as PSNR and LPIPS are saved in `metrics` tensorboard folder. 
* `random_seed`: The random seed that is used while randomly selecting test and train images from dataset. If you want to re-run an experiment, make sure to choose the same random seed.
* `save_dataset_filenames`: If you want to save the filenames of the data that you use while training or testing, set this flag to True; otherwise, set it to Fasle. The used train data will be saved in `train_filenames.txt` in the experiment directory in the format of:
```
data-root: <YOUR_DATA_ROOT>
source1: <PATH_TO_SOURCE1>
target1: <PATH_TO_TARGET1>
...
```
The used test data will be also be saved in `test_filenames.txt` in the experiment directory in the same format. This flag automatically deletes the previous files with the name `[train, test]_filenames.txt`. Make sure to set this flag to False if you don't want to save the image paths; otherwise, it will store a lot of data.
* `dataset_load_from_txt`: If you set this flag to True, the training happens on the train and test images from the `.txt` files that you provide in `train_load_from_filename` and `test_load_from_filename` respectively. 
* `experiment_dir`: Your experiment results be will saved in this main directory under `experiment_dir/runs/experiment_name`.
* `checkpoint_freq`: You can sepcify the frequency of storing the the model checkpoints with this flag. The model saves the checkpoints each `checkpoint_freq` epochs.
* `data_root`: You can choose the root of your data in this flag. For example, if you want to run an experiment on the per_person dataset, put `data_root: /video-conf/scratch/pantea/video_conf_datasets/per_person_dataset`. 
* `output_segmentation`: If you want to enable computing the predicted image's segmentation set this flag to True. You will be able to see the segmentation in the saved images.
* `emb_apply_masks`: If you want the embedding network to use the segmentation mask, set this variable to True. 
* `frame_num_from_paper`: If you want to use the paper's approach in selecting the train and test images, set this variable to True. If you set this variable to `False`, the source and the target images are randomly picked from all the sessions of one video.   
* `losses_test`: You can choose what losses to compute when testing the model; for example:  `--losses_test 'lpips, csim'`.
* `metrics`: You can choose what metrics you want to store; for example:  `--metrics 'PSNR, lpips, pose_matching_metric'`.
* `networks_test`: Order of forward passes during the training of gen (or gen and dis for sim sgd).
* `networks_train`: Order of forward passes during testing.
* `num_epochs`: Number of epochs to train the model
* `output_stickmen`: If you set to true, you can see the visualized keypoints.
* `runner_name`: The runner file that loads the networks and trains them in order.
* `test_freq`: The frequency of testing the model on test data.
* `visual_freq`: The frequency of storing the visual results.
* `init_experiment_dir`: If you want to train your model from some specific checkpoint, you should set this flag to point to the directory of the experiment that you want to use for initialization of the networks. This directory should have a `checkpoints` folder in it.  
* `init_networks`: This is the list of the networks you want to initialize with previous checkpoints. 
* `init_which_epoch`: The epoch to initialize the wights from.
* `which_epoch`: Epoch to continue training from, you can set the value to 1 when you want to train the network from scratch.
* `skip_test`: If set to False, the model automatically is tested on the test data and the results will be available in `experiment_dir/runs/experiment_name/images/test`.
* `frozen_networks`: If you want to freeze some networks, you can put their name in this list.


## Results folder

Your expeiment results will be stored in `experiment_dir/runs/experiment_name` directory. In this directory, you can find the model checkpoints in `checkpoints` folder in the format of `<EPOCH_NUMBER>_<NETWORK_NAME>.pth`. These checkpoints can be later used for initializing another network or for inference.
You can see the test and train images in `images` folder. 
We currently have two tensorboards in `metrics` folder and `tensorboard_paper` folder. 
The experiment's arguments are stored in `args.txt` and the losses are stored in `losses.pkl`. 


## Tensorboard
The tensorboard results are stored in two `tensorboard_paper` and `metrics` folders in `experiment_dir/runs/experiment_name` directory. They and can be viewed using tensorboard (on a browser at the reported port after running this command).
```bash
tensorboard --bind_all --logdir=<PATH_TO_TENSORBOARD>
```
After that you can view the results in your brower by the following urls:

```
Chunky: http://chunky.csail.mit.edu:<TENSORBOARD_PORT>/
Mapmaker: http://128.30.198.25:<TENSORBOARD_PORT>/
```

## Inference

For inference, you can currently use `examples/infer_test.py` file. Change the followings in the file to generate a new predicted target image:

* `experiment_name`: The name of the experiment that you want to test
* `experiment_dir` : The root of experiments
* `which_epoch`: The epoch that you want to test
* `preprocess`: If you want to preprocess two images, put this to `True`, if not the code will use preprocessed images and keypoints. 
* `source_img_path`: Path to your source image. Make sure to set it when you use `preprocess = True`.
* `target_img_path`: Path to your target image. Make sure to set it when you use `preprocess = True`.

If you set `preprocess` and `from_video` to True, you will use two frames from a video as source and target images. You need to change the following variables:
* `video_path` : Path to the video (Example: `'/video-conf/scratch/pantea/temp_dataset/id00012/_raOc3-IRsw/00110.mp4'`)
* `source_frame_num` : The frame number of the source  (Example: `0`)
* `target_frame_num` : The frame number of the target  (Example: `10`)

If you set `preprocess` to True and `from_video` to False, you will use to images as source and target imges. You need to define these paths like:
* `source_img_path` : Full path to the source image (Example: `'/home/pantea/NETS/nets_implementation/original_bilayer/examples/images/video_imgs/train_frames/0.jpg'`)
* `target_img_path` : Full path to the source image (Example: `'/home/pantea/NETS/nets_implementation/original_bilayer/examples/images/video_imgs/train_frames/1.jpg'`)

If you set `preprocess` and `from_video` to False, you will load the images, keypoints, and segmentations from stored datasets: 
* `dataset_root` : The dataset root (Example: `'/video-conf/scratch/pantea/temp_extracts'`)
* `source_relative_path` : The source image's relative path to dataset_root/imgs (Example: `'train/id00012/_raOc3-IRsw/00110/0'`)
* `target_relative_path` : The target image's relative path to dataset_root/imgs (Example: `'train/id00012/_raOc3-IRsw/00110/1'`)

```

+------------+------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| preprocess | from_video |                                                        Source & Target                                                                       |
+============+============+==============================================================================================================================================+
|            |    True    |  Picks two frames (source_frame_num and target_frame_num) from video in video_path preprocess them to find the keypoints                     |
|   True     |============+==============================================================================================================================================+
|            |   False    |  Picks the images in source_img_path and target_img_path and preprocess them to find the keypoints                                           |
+------------+------------+----------------------------------------------------------------------------------------------------------------------------------------------+
|            |    True    |  Not applicable                                                                                                                              |
|   False    |============+==============================================================================================================================================+
|            |   False    |  Loads preprocessed and save keypoints, images, and segmentations from dataset_root/[imgs, keypoints, segs]/{source or target}_relative_path |
+------------+------------+----------------------------------------------------------------------------------------------------------------------------------------------+
```
