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

## Tensorboard
The tensorboard results are stored in two `tensorboard_paper` and `metrics` folders in `experiment_dir/runs/experiment_name` directory. They and can be viewed using tensorboard (on a browser at the reported port after running this command).
```bash
tensorboard --bind_all --logdir=<PATH_TO_TENSORBOARD>
```
