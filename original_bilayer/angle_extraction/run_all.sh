#!/bin/bash

# 
# This script runs the extract_angles.py in separate tmux sessions in order to 
# parallelize them. You can see extract_angles.py for the paths I used for the model snapshot.
# The specific command I run is 
# ./run_all.sh "source ~/.bashrc" "/data/vision/billf/video-conf/scratch/pantea/temp_per_person_1_extracts" "/data/vision/billf/video-conf/scratch/vedantha/temp_1_angles" "/data/vision/billf/video-conf/scratch/vedantha/hope_weights/hopenet_robust_alpha1.pkl" 2 2
# Command to run is any code you want runnning when opening the tmux session like
# conda activate torch. If you don't want anything, just put a useless command there like ls.
#


command_to_run=${1}
dataset_path=${2}
save_path=${3}
model_path=${4}
num_gpus=${5}
num_threads_per_gpu=${6}
for (( c=0; c<$num_gpus; c++ ))
do  
	for (( k=0; k<$num_threads_per_gpu; k++ ))
	do  
		tmux new-session -d -s my_session_0 "${command_to_run} && python extract_angles.py --snapshot ${model_path} --gpu ${c} --proc $((num_gpus*num_threads_per_gpu)) --index $((c*num_threads_per_gpu + k)) --data_root ${dataset_path} --save_root ${save_path}"
	done
done
