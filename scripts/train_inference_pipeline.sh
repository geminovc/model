# This script train the Bilayer network with the flags in this script
# Then, the script picks the trained network and uses them in the inference pipeline
# The inference pipeline infers the predicted image on some unseen videos

data_root=/video-conf/scratch/pantea/per_person_1_three_datasets
root_to_yaws=/video-conf/scratch/pantea/pose_results/yaws/per_person_1_three_datasets/angles
num_epochs=6000
nets_repo=/home/pantea/NETS/nets_implementation
experiment_dir=/data/pantea/replace_Gtex_output_with_source/checkpoints
experiment_logs=/data/pantea/replace_Gtex_output_with_source/logs
replace_Gtex_output_with_source='True'

for wpr_loss_weight in 0.01 0.1 1
do
    experiments_name=per_person_yaw_wpr_loss_${wpr_loss_weight}
    frozen_networks=' '
    unfreeze_texture_generator_last_layers='True'
    unfreeze_inference_generator_last_layers='True'

    cd ${nets_repo}/original_bilayer/scripts/automated
    
    CUDA_VISIBLE_DEVICES=0 ./train_script.sh  ${experiments_name}  "from_bilayer" \
    "per_person" 2 ${num_epochs} 100 100 1000 100 'yaw' ${data_root} ${root_to_yaws} \
    "${frozen_networks}" "${unfreeze_texture_generator_last_layers}" "${unfreeze_inference_generator_last_layers}" \
    ${experiment_dir} ${wpr_loss_weight} "${replace_Gtex_output_with_source}"

    cd ${nets_repo}/original_bilayer/examples


    for video_id in  V4nIKszy_gc V9mbKUqFx0o  M_u0SV9wLro W4RJtdXRz-c fd_mtL88o1k
    do
        video_path=$(ls /video-conf/vedantha/voxceleb2/dev/mp4/id00015/${video_id} | sort -V | tail -n 1)
        save_dir=${experiment_logs}/${experiments_name}/images/${video_id}
        source_relative_path=unseen_test/id00015/${video_id}/${video_path%.*}/0 
        target_relative_path=unseen_test/id00015/${video_id}/${video_path%.*}/10 
        
        cd ${nets_repo}/original_bilayer/examples

        python infer_image.py \
        --experiment_dir ${experiment_dir} \
        --experiment_name ${experiments_name} \
        --which_epoch 3000 \
        --dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
        --source_relative_path ${source_relative_path} \
        --target_relative_path ${target_relative_path} \
        --save_dir ${save_dir}/per_person_yaw
    done
done
