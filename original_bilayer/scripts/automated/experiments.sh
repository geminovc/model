# MAIN_DIR=../..
# machine=${1}
# experiment_name=${2}
# initialization=${3}
# dataset_name=${4}
# batch_size=${5}
# num_epochs=${6}
# test_freq=${7}
# metrics_freq=${8}
# train_dataloader_name={9}

# root to the voxceleb2 videos
video_root=${1}
# path to where to save the dataset
data_root=${2}

# Generate the datasets

## Extract the keypoints from the videos
cd MAIN_DIR/keypoints_segmentations_extraction
./extract.sh 'train' ${video_root} 1 ${data_root}

## Split the data into test/train/unseen_test
cd MAIN_DIR/generate_dataset
python train_test_unseen_test.py ${data_root}

cd MAIN_DIR/scripts/automated

# From base experiment
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "experiment"  "from_base" "per_person" 2 7000 100 100 'voxceleb2' ${data_root}

# From paper's released checkpoint experiment
# Voxeleb2 dataloader
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "experiment"  "from_paper" "per_person" 2 7000 100 100 'voxceleb2' ${data_root}
# Difficult pose dataloader
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "experiment"  "from_paper" "per_person" 2 7000 100 100 'difficult_pose' ${data_root}
# L2 Distance dataloader 
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "experiment"  "from_paper" "per_person" 2 7000 100 100 'l2_distance' ${data_root}
# Yaw dataloader 
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "experiment"  "from_paper" "per_person" 2 7000 100 100 'yaw' ${data_root}

# Debugging experiment
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "debug"       "from_paper" "per_person" 2 10 1 1 'voxceleb2' ${data_root}

# Per_person train/test/unseen_test 
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "unseen_test" "from_paper" "per_person" 2 7000 100 100 'voxceleb2' ${data_root}