# This script runs the freezing experiments and makes image and video strips accordingly


# This function produces the predicted images from bilayer, per_person without yaw, per_person with yaw,
# per_video without yaw, and per_video with yaw and attaches them in different strips for comparison
# Inputs: 
# difficult_pose_flag: included the predicted high-frequency image and texture in the strips
# per_video_flag: includes per_video with yaw and per_video without yaw predicted images to the strips 
# source_relative_path: relative path from dataset_root (after keypoints, imgs, and segs) to the source images 
# target_relative_path: target path from dataset_root (after keypoints, imgs, and segs) to the source images
make_images () {
    cd ${nets_repo}/original_bilayer/examples
    
    if  [ "$generate_bilayer_results" = true ]
    then 
        python infer_image.py \
        --experiment_dir '/video-conf/scratch/pantea/bilayer_paper_released' \
        --experiment_name 'vc2-hq_adrianb_paper_main' \
        --which_epoch 2225 \
        --dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
        --source_relative_path ${source_relative_path} \
        --target_relative_path ${target_relative_path} \
        --save_dir ${save_dir}/bilayer_${video_id} 
    fi
    
    if  [ "$generate_per_person_results" = true ]
    then 
        python infer_image.py \
        --experiment_dir '/data/pantea/pantea_experiments_chunky/per_person/from_paper' \
        --experiment_name 'original_frozen_Gtex_from_identical' \
        --which_epoch 3000 \
        --dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
        --source_relative_path ${source_relative_path} \
        --target_relative_path ${target_relative_path} \
        --save_dir ${save_dir}/per_person_yaw_${video_id} 
        
        python infer_image.py \
        --experiment_dir '/video-conf/scratch/pantea_experiments_mapmaker/per_person/from_paper' \
        --experiment_name 'more_yaws_baseline_voxceleb2_easy_diff_combo' \
        --which_epoch 2000 \
        --dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
        --source_relative_path ${source_relative_path} \
        --target_relative_path ${target_relative_path} \
        --save_dir ${save_dir}/per_person_voxceleb2_${video_id} 

    fi

    if  [ "$generate_bilayer_results" = true ] && [ "$generate_per_person_results" = true ] 
    then 
        ### Make a strip of images containing: source | target | bilayer prediction | per_person random source-target | per_person close source-target 
        ffmpeg -y -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
        -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
        -i ${save_dir}/bilayer_${video_id}/pred_target_imgs_False_False.png \
        -i ${save_dir}/per_person_voxceleb2_${video_id}/pred_target_imgs_False_False.png \
        -i ${save_dir}/per_person_yaw_${video_id}/pred_target_imgs_False_False.png \
        -filter_complex "[0][1][2][3][4]hstack=inputs=5" ${save_dir}/stacked_per_person_with_yaw.png

        ### Make a strip of images containing: source | target | bilayer prediction | per_person random source-target 
        ffmpeg -y -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
        -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
        -i ${save_dir}/bilayer_${video_id}/pred_target_imgs_False_False.png \
        -i ${save_dir}/per_person_voxceleb2_${video_id}/pred_target_imgs_False_False.png \
        -filter_complex "[0][1][2][3]hstack=inputs=4" ${save_dir}/stacked_per_person_no_yaw.png
    

        if  [ "$difficult_pose_flag" = true ] 
        then 

            for current_model in bilayer per_person_voxceleb2 per_person_yaw
            do
                ### Make a strip of images containing: source | target | ${current_model} prediction | ${current_model} pred_tex_hf_rgbs | ${current_model} predicted target
                ffmpeg -y -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
                -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
                -i ${save_dir}/${current_model}_${video_id}/pred_tex_hf_rgbs_False_False.png \
                -i ${save_dir}/${current_model}_${video_id}/pred_target_imgs_False_False.png \
                -filter_complex "[0][1][2][3]hstack=inputs=4" ${save_dir}/${current_model}_${video_id}_with_pred_tex.png

                ### Make a strip of images containing: source | target | ${current_model} prediction | ${current_model} pred_target_delta_hf_rgbs | ${current_model} predicted target
                ffmpeg -y -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
                -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
                -i ${save_dir}/${current_model}_${video_id}/pred_target_delta_hf_rgbs_False_False.png \
                -i ${save_dir}/${current_model}_${video_id}/pred_target_imgs_False_False.png \
                -filter_complex "[0][1][2][3]hstack=inputs=4" ${save_dir}/${current_model}_${video_id}_with_pred_hf.png
            done

            ### Make a strip of images containing: source | target | per_person random source-target | per_person close source-target 
            ffmpeg -y -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
            -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
            -i ${save_dir}/per_person_voxceleb2_${video_id}/pred_target_imgs_False_False.png \
            -i ${save_dir}/per_person_yaw_${video_id}/pred_target_imgs_False_False.png \
            -filter_complex "[0][1][2][3]hstack=inputs=4" ${save_dir}/close_random_compare.png
        fi
    fi


    if  [ "$per_video_flag" = true ] 
    then 
        python infer_image.py \
        --experiment_dir ${experiment_dir} \
        --experiment_name ${experiments_name}_voxceleb_${video_id} \
        --which_epoch ${num_epochs} \
        --dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
        --source_relative_path ${source_relative_path} \
        --target_relative_path ${target_relative_path} \
        --save_dir ${save_dir}/per_video_voxceleb2_${video_id} 

        python infer_image.py \
        --experiment_dir ${experiment_dir} \
        --experiment_name ${experiments_name}_yaw_${video_id} \
        --which_epoch ${num_epochs} \
        --dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
        --source_relative_path ${source_relative_path} \
        --target_relative_path ${target_relative_path} \
        --save_dir ${save_dir}/per_video_yaw_${video_id} 

        ### Make a strip of images containing: source | target | per_video close prediction
        ffmpeg -y -i ${save_dir}/per_video_yaw_${video_id}/masked_source_imgs_False_False.png \
        -i ${save_dir}/per_video_yaw_${video_id}/masked_target_imgs_False_False.png \
        -i ${save_dir}/per_video_yaw_${video_id}/pred_target_imgs_False_False.png \
        -filter_complex "[0][1][2]hstack=inputs=3" ${save_dir}/source_target_per_video_yaw.png

        ### Make a strip of images containing: source | target | per_video random prediction
        ffmpeg -y -i ${save_dir}/per_video_voxceleb2_${video_id}/masked_source_imgs_False_False.png \
        -i ${save_dir}/per_video_voxceleb2_${video_id}/masked_target_imgs_False_False.png \
        -i ${save_dir}/per_video_voxceleb2_${video_id}/pred_target_imgs_False_False.png \
        -filter_complex "[0][1][2]hstack=inputs=3" ${save_dir}/source_target_per_video_voxceleb2.png

        if  [ "$generate_bilayer_results" = true ] && [ "$generate_per_person_results" = true ] 
        then 

            ### Make a strip of images containing: source | target | Bilayer | per_person random source-target| per_person close source-target | per_video random source-target \
            ### per_video close source-target
            ffmpeg -y -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
            -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
            -i ${save_dir}/bilayer_${video_id}/pred_target_imgs_False_False.png \
            -i ${save_dir}/per_person_voxceleb2_${video_id}/pred_target_imgs_False_False.png \
            -i ${save_dir}/per_person_yaw_${video_id}/pred_target_imgs_False_False.png \
            -i ${save_dir}/per_video_voxceleb2_${video_id}/pred_target_imgs_False_False.png \
            -i ${save_dir}/per_video_yaw_${video_id}/pred_target_imgs_False_False.png \
            -filter_complex "[0][1][2][3][4][5][6]hstack=inputs=7" ${save_dir}/stacked_per_video_with_yaw.png

            ### Make a strip of images containing: source | target | Bilayer | per_person random source-target | per_video close source-target 
            ffmpeg -y -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
            -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
            -i ${save_dir}/bilayer_${video_id}/pred_target_imgs_False_False.png \
            -i ${save_dir}/per_person_voxceleb2_${video_id}/pred_target_imgs_False_False.png \
            -i ${save_dir}/per_video_voxceleb2_${video_id}/pred_target_imgs_False_False.png \
            -filter_complex "[0][1][2][3][4]hstack=inputs=5" ${save_dir}/stacked_per_video_no_yaw.png
        fi
    fi

} 


# This function produces the predicted video from bilayer, per_person without yaw, per_person with yaw,
# per_video without yaw, and per_video with yaw and attaches the videos in different order for comparison
# Inputs: 
# relative_path_base: relative path from dataset_root (after keypoints, imgs, and segs) to the session
make_videos () {
    cd ${nets_repo}/original_bilayer/examples

    for video_technique in representative_sources last_frame_next_source last_predicted_next_source
    do

        if  [ "$generate_bilayer_results" = true ]
        then 
            python infer_video.py \
            --experiment_dir '/video-conf/scratch/pantea/bilayer_paper_released' \
            --experiment_name 'vc2-hq_adrianb_paper_main' \
            --which_epoch 2225 \
            --video_path /video-conf/vedantha/voxceleb2/dev/mp4/id00015/${video_id}/${video_path} \
            --dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
            --relative_path_base ${relative_path_base} \
            --yaw_root '/video-conf/scratch/pantea/pose_results/yaws/per_person_1_three_datasets/angles' \
            --save_dir ${save_dir}/bilayer_${video_id} \
            --video_technique ${video_technique}
        fi

        if  [ "$generate_per_person_results" = true ]
        then 
            python infer_video.py \
            --experiment_dir '/video-conf/scratch/pantea_experiments_mapmaker/per_person/from_paper' \
            --experiment_name 'more_yaws_baseline_voxceleb2_easy_diff_combo' \
            --which_epoch 2000 \
            --video_path /video-conf/vedantha/voxceleb2/dev/mp4/id00015/${video_id}/${video_path} \
            --dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
            --relative_path_base ${relative_path_base} \
            --yaw_root '/video-conf/scratch/pantea/pose_results/yaws/per_person_1_three_datasets/angles' \
            --save_dir ${save_dir}/per_person_voxceleb2_${video_id} \
            --video_technique ${video_technique}
        
            python infer_video.py \
            --experiment_dir '/data/pantea/pantea_experiments_chunky/per_person/from_paper' \
            --experiment_name 'original_frozen_Gtex_from_identical' \
            --which_epoch 2000 \
            --video_path /video-conf/vedantha/voxceleb2/dev/mp4/id00015/${video_id}/${video_path} \
            --dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
            --relative_path_base ${relative_path_base} \
            --yaw_root '/video-conf/scratch/pantea/pose_results/yaws/per_person_1_three_datasets/angles' \
            --save_dir ${save_dir}/per_person_yaw_${video_id} \
            --video_technique ${video_technique}
        fi

        if  [ "$per_video_flag" = true ]
        then 
            python infer_video.py \
            --experiment_dir ${experiment_dir} \
            --experiment_name ${experiments_name}_yaw_${video_id} \
            --which_epoch ${num_epochs} \
            --video_path /video-conf/vedantha/voxceleb2/dev/mp4/id00015/${video_id}/${video_path} \
            --dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
            --relative_path_base ${relative_path_base}  \
            --yaw_root '/video-conf/scratch/pantea/pose_results/yaws/per_person_1_three_datasets/angles' \
            --save_dir ${save_dir}/per_video_yaw_${video_id} \
            --video_technique ${video_technique}

            python infer_video.py \
            --experiment_dir ${experiment_dir} \
            --experiment_name ${experiments_name}_voxceleb_${video_id} \
            --which_epoch ${num_epochs} \
            --video_path /video-conf/vedantha/voxceleb2/dev/mp4/id00015/${video_id}/${video_path} \
            --dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
            --relative_path_base ${relative_path_base}  \
            --yaw_root '/video-conf/scratch/pantea/pose_results/yaws/per_person_1_three_datasets/angles' \
            --save_dir ${save_dir}/per_video_voxceleb2_${video_id} \
            --video_technique ${video_technique}
        fi

        if  [ "$generate_bilayer_results" = true ] && [ "$generate_per_person_results" = true ] && [ "$per_video_flag" = true ]
        then 

            ### Make a strip of videos containing: target | Bilayer | per_person random source-target | per_person close source-target | per_video random source-target | per_video close source-target 
            ffmpeg -y -i ${save_dir}/per_video_yaw_${video_id}/masked_original.mp4 \
            -i ${save_dir}/bilayer_${video_id}/${video_technique}.mp4 \
            -i ${save_dir}/per_person_voxceleb2_${video_id}/${video_technique}.mp4 \
            -i ${save_dir}/per_person_yaw_${video_id}/${video_technique}.mp4 \
            -i ${save_dir}/per_video_voxceleb2_${video_id}/${video_technique}.mp4 \
            -i ${save_dir}/per_video_yaw_${video_id}/${video_technique}.mp4 \
            -filter_complex "[0][1][2][3][4][5]hstack=inputs=6" ${save_dir}/stacked_${video_technique}_all.mp4
        
            ### Make a strip of videos containing: target | Bilayer | per_person close source-target | per_video close source-target 
            ffmpeg -y -i ${save_dir}/per_video_yaw_${video_id}/masked_original.mp4 \
            -i ${save_dir}/bilayer_${video_id}/${video_technique}.mp4 \
            -i ${save_dir}/per_person_yaw_${video_id}/${video_technique}.mp4 \
            -i ${save_dir}/per_video_yaw_${video_id}/${video_technique}.mp4 \
            -filter_complex "[0][1][2][3]hstack=inputs=4" ${save_dir}/stacked_${video_technique}_with_yaw.mp4
            
            ### Make a strip of videos containing: target | Bilayer | per_video close source-target 
            ffmpeg -y -i ${save_dir}/per_video_yaw_${video_id}/masked_original.mp4 \
            -i ${save_dir}/bilayer_${video_id}/${video_technique}.mp4 \
            -i ${save_dir}/per_video_yaw_${video_id}/${video_technique}.mp4 \
            -filter_complex "[0][1][2]hstack=inputs=3" ${save_dir}/stacked_${video_technique}_orig_by_ours_with_yaw.mp4

        fi
    done
    
} 


run_per_video_experiments () {
    for video_id in  V4nIKszy_gc V9mbKUqFx0o  M_u0SV9wLro W4RJtdXRz-c fd_mtL88o1k
    do
        cd ${nets_repo}/original_bilayer/generate_dataset

        python make_new_per_video_dataset.py --target_video_id id00015/${video_id} --num_test_sessions 1 --data_root ${data_root} --yaw_root ${root_to_yaws} 
        
        cd ${nets_repo}/original_bilayer/scripts/automated
        

        CUDA_VISIBLE_DEVICES=0 ./train_script.sh  ${experiments_name}_voxceleb_${video_id}  "from_personalized" "per_video" 2 ${num_epochs} 20 20 20 20 'voxceleb2' ${data_root} ${root_to_yaws} "${frozen_networks}" "${unfreeze_texture_generator_last_layers}" "${unfreeze_inference_generator_last_layers}" ${experiment_dir} ${wpr_loss_weight} "${replace_Gtex_output_with_source}"

        CUDA_VISIBLE_DEVICES=0 ./train_script.sh  ${experiments_name}_yaw_${video_id}  "from_personalized" "per_video" 2 ${num_epochs} 20 20 20 20 'yaw' ${data_root} ${root_to_yaws} "${frozen_networks}" "${unfreeze_texture_generator_last_layers}" "${unfreeze_inference_generator_last_layers}" ${experiment_dir} ${wpr_loss_weight} "${replace_Gtex_output_with_source}"

        cd ${nets_repo}/original_bilayer/examples

        video_path=$(ls /video-conf/vedantha/voxceleb2/dev/mp4/id00015/${video_id} | sort -V | tail -n 1)

        generate_bilayer_results=true
        generate_per_person_results=true
        difficult_pose_flag=true
        per_video_flag=true
        save_dir=${experiment_logs}/${experiments_name}/images/${video_id}
        source_relative_path=unseen_test/id00015/${video_id}/${video_path%.*}/0 
        target_relative_path=unseen_test/id00015/${video_id}/${video_path%.*}/10 
        make_images

        ## unseen_test
        per_video_flag=true
        save_dir=${experiment_logs}/${experiments_name}/videos/${video_id}
        relative_path_base=unseen_test/id00015/${video_id}/${video_path%.*}
        make_videos 

    done
 
}

data_root=/data/pantea/datasets/per_video_freezing_dataset
root_to_yaws=/data/pantea/dataset_yaws/per_video_freezing_dataset_yaws/angles
num_epochs=60
experiment_dir=/data/pantea/freezing_per_video/checkpoints
nets_repo=/home/pantea/NETS/nets_implementation
experiment_logs=/data/pantea/freezing_per_video/logs
experiment_name_prefix=
#if with_frozen_last_layer_flag is set to true, experiments in which the last layer of G_inf or G_tex is frozen are conducted
with_frozen_last_layer_flag=false
#if single_last_layer_flag is set to true, experiments in which only the last layer of G_inf or G_tex is unfrozen are conducted
single_last_layer_flag=false


frozen_networks=' '
unfreeze_texture_generator_last_layers='True'
unfreeze_inference_generator_last_layers='True'
experiments_name=${experiment_name_prefix}no_frozen
run_per_video_experiments

frozen_networks='identity_embedder, keypoints_embedder'
unfreeze_texture_generator_last_layers='True'
unfreeze_inference_generator_last_layers='True'
experiments_name=${experiment_name_prefix}unfrozen_G_tex_and_G_inf_frozen_rest
run_per_video_experiments

frozen_networks="identity_embedder, keypoints_embedder, inference_generator"
unfreeze_texture_generator_last_layers='True'
unfreeze_inference_generator_last_layers='False'
experiments_name=${experiment_name_prefix}unfrozen_G_tex_frozen_rest
run_per_video_experiments

frozen_networks="identity_embedder, keypoints_embedder, inference_generator"
unfreeze_texture_generator_last_layers='True'
unfreeze_inference_generator_last_layers='True'
experiments_name=${experiment_name_prefix}unfrozen_G_tex_and_last_layer_of_G_inf_frozen_rest
run_per_video_experiments

frozen_networks="identity_embedder, keypoints_embedder, texture_generator"
unfreeze_texture_generator_last_layers='False'
unfreeze_inference_generator_last_layers='True'
experiments_name=${experiment_name_prefix}unfrozen_G_inf_frozen_rest
run_per_video_experiments

frozen_networks="identity_embedder, keypoints_embedder, texture_generator"
unfreeze_texture_generator_last_layers='True'
unfreeze_inference_generator_last_layers='True'
experiments_name=${experiment_name_prefix}unfrozen_G_inf_and_last_layer_of_G_tex_frozen_rest
run_per_video_experiments

if  [ "$with_frozen_last_layer_flag" = true ] 
then 

    frozen_networks='identity_embedder, keypoints_embedder'
    unfreeze_texture_generator_last_layers='False'
    unfreeze_inference_generator_last_layers='True'
    experiments_name=${experiment_name_prefix}unfrozen_G_tex_with_frozen_last_layer_and_G_inf_frozen_rest
    run_per_video_experiments

    frozen_networks='identity_embedder, keypoints_embedder'
    unfreeze_texture_generator_last_layers='True'
    unfreeze_inference_generator_last_layers='False'
    experiments_name=${experiment_name_prefix}unfrozen_G_tex_and_G_inf_with_frozen_last_layer_frozen_rest
    run_per_video_experiments

    frozen_networks='identity_embedder, keypoints_embedder'
    unfreeze_texture_generator_last_layers='False'
    unfreeze_inference_generator_last_layers='False'
    experiments_name=${experiment_name_prefix}unfrozen_G_tex_with_frozen_last_layer_and_G_inf_with_frozen_last_layer_frozen_rest
    run_per_video_experiments

    frozen_networks="identity_embedder, keypoints_embedder, inference_generator"
    unfreeze_texture_generator_last_layers='False'
    unfreeze_inference_generator_last_layers='False'
    experiments_name=${experiment_name_prefix}unfrozen_G_tex_with_frozen_last_layer_frozen_rest
    run_per_video_experiments

    frozen_networks="identity_embedder, keypoints_embedder, inference_generator"
    unfreeze_texture_generator_last_layers='False'
    unfreeze_inference_generator_last_layers='True'
    experiments_name=${experiment_name_prefix}unfrozen_G_tex_with_frozen_last_layer_and_last_layer_of_G_inf_frozen_rest
    run_per_video_experiments

    frozen_networks="identity_embedder, keypoints_embedder, texture_generator"
    unfreeze_texture_generator_last_layers='False'
    unfreeze_inference_generator_last_layers='False'
    experiments_name=${experiment_name_prefix}unfrozen_G_inf_with_frozen_last_layer_frozen_rest
    run_per_video_experiments

    frozen_networks="identity_embedder, keypoints_embedder, texture_generator"
    unfreeze_texture_generator_last_layers='False'
    unfreeze_inference_generator_last_layers='True'
    experiments_name=${experiment_name_prefix}unfrozen_G_inf_with_frozen_last_layer_and_last_layer_of_G_tex_frozen_rest
    run_per_video_experiments


fi

if  [ "$single_last_layer_flag" = true ] 
then 

    frozen_networks="identity_embedder, keypoints_embedder, texture_generator, inference_generator"
    unfreeze_texture_generator_last_layers='False'
    unfreeze_inference_generator_last_layers='True'
    experiments_name=${experiment_name_prefix}unfrozen_last_layer_G_inf_frozen_rest
    run_per_video_experiments

    frozen_networks="identity_embedder, keypoints_embedder, texture_generator, inference_generator"
    unfreeze_texture_generator_last_layers='True'
    unfreeze_inference_generator_last_layers='False'
    experiments_name=${experiment_name_prefix}unfrozen_last_layer_G_tex_frozen_rest
    run_per_video_experiments

    frozen_networks="identity_embedder, keypoints_embedder, texture_generator, inference_generator"
    unfreeze_texture_generator_last_layers='True'
    unfreeze_inference_generator_last_layers='True'
    experiments_name=${experiment_name_prefix}unfrozen_last_layer_G_tex_and_last_layer_G_inf_frozen_rest
    run_per_video_experiments

fi
