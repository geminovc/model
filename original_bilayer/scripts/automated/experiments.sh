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


# Debugging experiment
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "debug" "from_base" "per_person" 2 7000 1 1 False False 0.1 False
