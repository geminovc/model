# This script produces the predicted images from bilayer, per_person without yaw, per_person with yaw,
# per_video without yaw, and per_video with yaw and attaches them in different strips for comparison
# Inputs: 
# difficult_pose_flag: included the predicted high-frequency image and texture in the strips
# per_video_flag: includes per_video with yaw and per_video without yaw predicted images to the strips 
# source_relative_path: relative path from dataset_root (after keypoints, imgs, and segs) to the source images 
# target_relative_path: target path from dataset_root (after keypoints, imgs, and segs) to the source images

source_relative_path=$1
target_relative_path=$2
save_dir=$3
video_id=$4
generate_bilayer_results=$5
generate_per_person_results=$6
difficult_pose_flag=$7
per_video_flag=$8
nets_repo=$9
experiment_dir=${10}
experiments_name=${11}
num_epochs=${12}

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



