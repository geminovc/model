# root to the voxceleb2 videos
video_root=${1}
# path to where to save the dataset
data_root=${2}
# path to save yaws
yaw_root=${3}
# Gpu
gpu_id=${4}

MAIN_DIR=../..

# Generate the datasets

## Extract the keypoints from the videos
cd MAIN_DIR/keypoints_segmentations_extraction
./extract.sh 'train' ${video_root} 1 ${data_root}

## Split the data into test/train/unseen_test
cd MAIN_DIR/generate_dataset
python train_test_unseen_test.py ${data_root}

## Extract the yaws for the dataset
cd MAIN_DIR/angle_extraction
python extract_angles.py \
--snapshot /video-conf/scratch/vedantha/hope_weights/hopenet_robust_alpha1.pkl \
--gpu ${gpu_id} \
--data_root ${data_root} \
--save_root ${yaw_root}

root_to_yaws=yaw_root/angles

# Run the experiments
cd MAIN_DIR/scripts/automated

## From base experiment
CUDA_VISIBLE_DEVICES=${gpu_id} ./train_script.sh  "chunky" "experiment"  "from_base" "per_person" 2 7000 100 100 1000 500 'voxceleb2' ${data_root} ${root_to_yaws}

## From paper's released checkpoint experiment

## Voxeleb2 dataloader
CUDA_VISIBLE_DEVICES=${gpu_id} ./train_script.sh  "chunky" "experiment"  "from_paper" "per_person" 2 7000 100 100 1000 500 'voxceleb2' ${data_root} ${root_to_yaws}

## Yaw dataloader 
CUDA_VISIBLE_DEVICES=${gpu_id} ./train_script.sh  "chunky" "experiment"  "from_paper" "per_person" 2 7000 100 100 1000 500 'yaw' ${data_root} ${root_to_yaws}

# Debugging experiment
CUDA_VISIBLE_DEVICES=${gpu_id} ./train_script.sh  "chunky" "debug"       "from_paper" "per_person" 2 10 1 1 1 1 'voxceleb2' ${data_root} ${root_to_yaws}

# Per_person train/test/unseen_test 
CUDA_VISIBLE_DEVICES=${gpu_id} ./train_script.sh  "chunky" "unseen_test" "from_paper" "per_person" 2 7000 100 100 1000 500 'voxceleb2' ${data_root} ${root_to_yaws}
