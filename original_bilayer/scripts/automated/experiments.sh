# machine=${1}
# experiment_name=${2}
# initialization=${3}
# dataset_name=${4}
# batch_size=${5}
# num_epochs=${6}
# test_freq=${7}
# metrics_freq=${8}
# augment_with_general=${9}
# sample_general_dataset=${10}
# augment_with_general_ratio=${11}
# inf_apply_masks=${12}

# From base experiment
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "no_augmentation_no_mask"   "from_base" "per_person" 2 7000 1 1 False False 0.1 False
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "no_augmentation_with_mask" "from_base" "per_person" 2 7000 1 1 False False 0.1 True

# From paper's released checkpoint experiment
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "no_augmentation_no_mask"   "from_paper" "per_person" 2 7000 1 1 False False 0.1 False
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "no_augmentation_with_mask" "from_paper" "per_person" 2 7000 1 1 False False 0.1 True
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "augmented_no_sampling_0.6_no_mask"      "from_paper" "per_person" 2 7000 1 1 True False 0.6 False

# Debugging experiment
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "debug" "from_base" "per_person" 2 7000 1 1 False False 0.1 False
