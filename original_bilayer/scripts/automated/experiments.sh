# MAIN_DIR=../..
# machine=${1}
# experiment_name=${2}
# initialization=${3}
# dataset_name=${4}
# batch_size=${5}
# num_epochs=${6}
# test_freq=${7}
# metrics_freq=${8}
# dataloader_name={9}

# From base experiment
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "experiment"  "from_base" "per_person" 2 7000 100 100 'voxceleb2'

# From paper's released checkpoint experiment
# Voxeleb2 dataloader
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "experiment"  "from_paper" "per_person" 2 7000 100 100 'voxceleb2'
# Difficult pose dataloader
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "experiment"  "from_paper" "per_person" 2 7000 100 100 'difficult_pose'
# L2 Distance dataloader 
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "experiment"  "from_paper" "per_person" 2 7000 100 100 'l2_distance'
# Yaw dataloader 
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "experiment"  "from_paper" "per_person" 2 7000 100 100 'yaw'

# Debugging experiment
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "debug"       "from_paper" "per_person" 2 10 1 1 'voxceleb2'

# Per_person train/test/unseen_test 
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "unseen_test" "from_paper" "per_person" 2 7000 100 100 'voxceleb2'