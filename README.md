# Running the bilayer model

## Setup
### Conda Environment
Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and 
initialize your shell by running `conda init <YOUR_SHELL_NAME>`.

Clone and setup new environment, check that it has been created and activate it.
```bash
conda env create -f source_bilayer/environment.yml
conda env list
conda activate bilayer
```

### Pre-trained Weights 
Download the `pretrained-weights` folder from [here](https://drive.google.com/drive/folders/11SwIYnk3KY61d8qa17Nlb0BN9j57B3L6) and place it inside `source_bilayer/bilayer-model`.
The pre-trained weights are courtesy [Fast Bi-layer Neural Synthesis of One-Shot Realistic Head Avatars](https://arxiv.org/abs/2008.10174) by Zakharov et. al.

### Videos and Frames
The pipeline expects a series of frames from an original video. 
First download a sample video from [here](https://www.youtube.com/watch?v=gEDChDOM1_U&vl=en)
```bash
cd source_bilayer/bilayer-model
sudo apt-get install youtube-dl
youtube-dl https://www.youtube.com/watch?v=gEDChDOM1_U&vl=en -o sundar_pichai.mp4
```

The model expects the frames to be located in a directory organization similar to [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html). To match that, create two subdirectories inside a main directory before saving the frames of the video.
```bash
mkdir -p mp4/sundar/frames
python save_video_as_images.py sundar_pichai.mp4 mp4/sundar/frames/
```

## Training 
```bash
cd examples
python train_tmux.py
```
This creates a new model for the first run. On additional runs, it uses the model saved at the end of the epoch specified by the `begin` parameter. 

Runs are labeled according to the `label_run` parameter and can be viewed using tensorboard (on a browser at port 6006).
```bash
tensorboard --bind_all --logdir=runs/<YOUR_RUN_NAME>
```
