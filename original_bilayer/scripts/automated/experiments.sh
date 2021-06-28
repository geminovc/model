# machine=${1}
# experiment_name=${2}
# initialization=${3}
# dataset_name=${4}
# batch_size=${5}
# num_epochs=${6}
# test_freq=${7}
# metrics_freq=${8}
# use_dropout=${9}
# dropout_networks={10}

# From base experiments

# From base, without dropout
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "no_dropout" "from_base" "per_person" 2 7000 100 100 False
# From base, with dropout
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "dropout"    "from_base" "per_person" 2 7000 100 100 True

# From paper's released checkpoint experiment

# From paper, without dropout
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "no_dropout" "from_paper" "per_person" 2 7000 100 100 False
# From paper, with dropout in G_tex
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "dropout"    "from_paper" "per_person" 2 7000 100 100 True 'texture_generator: 0.5'
# From paper, with dropout in G_inf
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "dropout"    "from_paper" "per_person" 2 7000 100 100 True 'inference_generator: 0.5'
# From paper, with dropout in G_tex and G_inf
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "dropout"    "from_paper" "per_person" 2 7000 100 100 True 'inference_generator: 0.5, texture_generator: 0.5'


# Debugging experiment 
CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" "debug" "from_base" "per_person" 2 7000 1 1 True 'texture_generator: 0.5'
