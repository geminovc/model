#!/bin/bash

gpu_index=$1
base_checkpoint=$2
netadapt_checkpoint=$3
log_dir=$4

# Run the command using the specified GPU index and file paths
for person_id in "seth_meyers" "kayleigh" "needle_drop" "trevor_noah" "jen_psaki"
do
    if [ $# -eq 5 ]
    then
        CONV_TYPE=depthwise \
        CUDA_VISIBLE_DEVICES=$gpu_index \
        python run.py \
        --config config/paper_configs/netadapt/finetune.yaml \
        --checkpoint "$base_checkpoint" \
        --netadapt_checkpoint "$netadapt_checkpoint" \
        --person_id "$person_id" \
        --experiment_name "$person_id" \
        --log_dir "$log_dir"
    else
        CUDA_VISIBLE_DEVICES=$gpu_index \
        python run.py \
        --config config/paper_configs/netadapt/finetune.yaml \
        --checkpoint "$base_checkpoint" \
        --netadapt_checkpoint "$netadapt_checkpoint" \
        --person_id "$person_id" \
        --experiment_name "$person_id" \
        --log_dir "$log_dir"
    fi
done

