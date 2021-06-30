#!/bin/bash

# 
# This script runs the extract_angles.py in separate tmux sessions in order to 
# parallelize them. You can see extract_angles.py for the paths I used for the model snapshot.
# Command to run is any code you want runnning when opening the tmux session like
# conda activate torch or something.
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
		tmux new-session -d -s my_session_0 "${command_to_run} && python extract_angles.py --snapshot ${model_path}--gpu ${c} --index $((c*num_threads_per_gpu + k)) --root ${dataset_path} --save_path ${save_path}"
	done
done
