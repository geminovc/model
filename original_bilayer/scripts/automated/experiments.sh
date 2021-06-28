# MAIN_DIR=../..
# machine=${1}
# experiment_name=${2}
# initialization=${3}
# dataset_name=${4}
# batch_size=${5}
# num_epochs=${6}
# test_freq=${7}
# metrics_freq=${8}
# mask_source_target=${9}

# From base experiment
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "no_augmentation_no_mask"   "from_base" "per_person" 2 7000 100 100 False
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "no_augmentation_with_mask" "from_base" "per_person" 2 7000 100 100 True

# From paper's released checkpoint experiment
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "no_augmentation_no_mask"   "from_paper" "per_person" 2 7000 100 100 False
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "no_augmentation_with_mask" "from_base"  "per_person" 2 7000 100 100 True


# Debugging experiment
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "debug" "from_paper" "per_person" 2 10 1 1 True