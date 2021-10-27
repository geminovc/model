# This script runs the per_video experiments
# (fine-uning the Bilayer on a single video) with also trying freezing different parts of the Bilayer network
# After training, the script uses inference piepline to generate predicted unseen image and video session strips accordingly


run_per_video_experiments () {
    frozen_networks="$1"
    unfreeze_texture_generator_last_layers="$2"
    unfreeze_inference_generator_last_layers="$3"
    experiments_name=$4
    
    for video_id in  V4nIKszy_gc V9mbKUqFx0o  M_u0SV9wLro W4RJtdXRz-c fd_mtL88o1k
    do 

        if  [ "$do_training" = true ]
        then
            cd ${nets_repo}/original_bilayer/generate_dataset
            python make_new_per_video_dataset.py --target_video_id id00015/${video_id} \
            --num_test_sessions 1 --data_root ${data_root} --yaw_root ${root_to_yaws}

            cd ${nets_repo}/original_bilayer/scripts/automated

            CUDA_VISIBLE_DEVICES=0 ./train_script.sh  ${experiments_name}_voxceleb_${video_id}  \
            "from_personalized" "per_video" 2 ${num_epochs} 20 20 20 20 'voxceleb2' ${data_root} \
            ${root_to_yaws} "${frozen_networks}" "${unfreeze_texture_generator_last_layers}" \
            "${unfreeze_inference_generator_last_layers}" ${experiment_dir} ${wpr_loss_weight} \
            "${replace_Gtex_output_with_source}"

            CUDA_VISIBLE_DEVICES=0 ./train_script.sh  ${experiments_name}_yaw_${video_id} \
            "from_personalized" "per_video" 2 ${num_epochs} 20 20 20 20 'yaw' ${data_root} \
            ${root_to_yaws} "${frozen_networks}" "${unfreeze_texture_generator_last_layers}" \
            "${unfreeze_inference_generator_last_layers}" ${experiment_dir} ${wpr_loss_weight} \
            "${replace_Gtex_output_with_source}"

        fi
        cd ${nets_repo}/scripts

        video_path=$(ls /video-conf/vedantha/voxceleb2/dev/mp4/id00015/${video_id} | sort -V | tail -n 1)

        generate_bilayer_results=true
        generate_per_person_results=true
        difficult_pose_flag=true
        per_video_flag=true
        save_dir=${experiment_logs}/${experiments_name}/images/${video_id}
        source_relative_path=unseen_test/id00015/${video_id}/${video_path%.*}/0 
        target_relative_path=unseen_test/id00015/${video_id}/${video_path%.*}/10 

        ./make_images.sh $source_relative_path $target_relative_path $save_dir $video_id \
        $generate_bilayer_results $generate_per_person_results $difficult_pose_flag \
        $per_video_flag $nets_repo $experiment_dir $experiments_name $num_epochs

        ## unseen_test
        per_video_flag=true
        save_dir=${experiment_logs}/${experiments_name}/videos/${video_id}
        relative_path_base=unseen_test/id00015/${video_id}/${video_path%.*}
        ./make_videos.sh $relative_path_base $save_dir $video_id $generate_bilayer_results \
        $generate_per_person_results $difficult_pose_flag $per_video_flag $nets_repo $experiment_dir \
        $experiments_name $num_epochs

    done
 
}

data_root=/data/pantea/datasets/per_video_freezing_dataset
root_to_yaws=/data/pantea/dataset_yaws/per_video_freezing_dataset_yaws/angles
num_epochs=60
wpr_loss_weight=0.1
replace_Gtex_output_with_source='False'
experiment_dir=/data/pantea/Experiments/per_video_freezing/per_video_freezing_checkpoints/per_video/from_paper
nets_repo=/home/pantea/NETS/nets_implementation
experiment_logs=/data/pantea/Experiments/per_video_freezing/per_video_freezing_logs
experiment_name_prefix=
do_training=false
#if with_frozen_last_layer_flag is true, experiments in which the last layer of G_inf or G_tex is frozen are conducted
with_frozen_last_layer_flag=false
#if single_last_layer_flag is set to true, experiments in which only the last layer of G_inf or G_tex is unfrozen are conducted
single_last_layer_flag=false


# The following experiments try freezing different parts of the Bilayer networks and train on a single video
# The frist argument is the name of the networks that are frozen
# The second argument is a boolan and unfreezes the texture generator's last layers (after AdaSpade layer) if True
# The third argument is a boolan and unfreezes the inference generator's last layers (after AdaSpade layer) if True
# The fourth argument is the name of the experiment

run_per_video_experiments ' ' 'True' 'True' ${experiment_name_prefix}no_frozen

run_per_video_experiments "identity_embedder, keypoints_embedder" \
'True' 'True' ${experiment_name_prefix}unfrozen_G_tex_and_G_inf

frozen_networks_without_Gtex="identity_embedder, keypoints_embedder, inference_generator"
frozen_networks_without_Ginf="identity_embedder, keypoints_embedder, texture_generator"

run_per_video_experiments "$frozen_networks_without_Gtex" 'True' 'False' \
${experiment_name_prefix}unfrozen_G_tex

run_per_video_experiments "$frozen_networks_without_Gtex" 'True' 'True' \
${experiment_name_prefix}unfrozen_G_tex_and_last_layer_of_G_inf

run_per_video_experiments "$frozen_networks_without_Ginf" 'False' 'True' \
${experiment_name_prefix}unfrozen_G_inf

run_per_video_experiments "$frozen_networks_without_Ginf" 'True' 'True' \
${experiment_name_prefix}unfrozen_G_inf_and_last_layer_of_G_tex

if  [ "$with_frozen_last_layer_flag" = true ] 
then 

    run_per_video_experiments 'identity_embedder, keypoints_embedder' 'False' 'True' \
    ${experiment_name_prefix}unfrozen_G_tex_with_frozen_last_layer_and_G_inf

    run_per_video_experiments 'identity_embedder, keypoints_embedder' 'True' 'False' \
    ${experiment_name_prefix}unfrozen_G_tex_and_G_inf_with_frozen_last_layer

    run_per_video_experiments 'identity_embedder, keypoints_embedder' 'False' 'False' \
    ${experiment_name_prefix}unfrozen_G_tex_with_frozen_last_layer_and_G_inf_with_frozen_last_layer
    
    run_per_video_experiments "$frozen_networks_without_Gtex" 'False' 'False' \
    ${experiment_name_prefix}unfrozen_G_tex_with_frozen_last_layer
    
    run_per_video_experiments "$frozen_networks_without_Gtex" 'False' 'True' \
    ${experiment_name_prefix}unfrozen_G_tex_with_frozen_last_layer_and_last_layer_of_G_inf
    
    run_per_video_experiments "$frozen_networks_without_Ginf" 'False' 'False' \
    ${experiment_name_prefix}unfrozen_G_inf_with_frozen_last_layer
    
    run_per_video_experiments "$frozen_networks_without_Ginf" 'False' 'True' \
    ${experiment_name_prefix}unfrozen_G_inf_with_frozen_last_layer_and_last_layer_of_G_tex

fi

if  [ "$single_last_layer_flag" = true ] 
then 
    all_networks="identity_embedder, keypoints_embedder, texture_generator, inference_generator"
    run_per_video_experiments "$all_networks" 'False' 'True' \
    ${experiment_name_prefix}unfrozen_last_layer_G_inf
    
    run_per_video_experiments "$all_networks" 'True' 'False' \
    ${experiment_name_prefix}unfrozen_last_layer_G_tex
    
    run_per_video_experiments "$all_networks" 'True' 'True' \
    ${experiment_name_prefix}unfrozen_last_layer_G_tex_and_last_layer_G_inf

fi
