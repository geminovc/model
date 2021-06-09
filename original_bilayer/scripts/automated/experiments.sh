# MAIN_DIR=../..
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
# inf_apply_masks=${11}


#
CUDA_VISIBLE_DEVICES=2 ./train_from_paper_checkpoints.sh  "chunky" "debug_dd" "from_base" "per_person" 2 7000 1 1 False False False
