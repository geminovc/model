# MAIN_DIR=../..
# machine=${1}
# experiment_name=${2}
# initialization=${3}
# dataset_name=${4}
# batch_size=${5}
# num_epochs=${6}
# test_freq=${7}
# metrics_freq=${8}


# Per_person train/test/unseen_test 
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "unseen_test"  "from_paper" "per_person" 2 7000 100 100 