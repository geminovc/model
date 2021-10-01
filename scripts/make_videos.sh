# This function produces the predicted video from bilayer, per_person without yaw, per_person with yaw,
# per_video without yaw, and per_video with yaw and attaches the videos in different order for comparison
# Inputs: 
# relative_path_base: relative path from dataset_root (after keypoints, imgs, and segs) to the session

relative_path_base=$1
save_dir=$2
video_id=$3
generate_bilayer_results=$4
generate_per_person_results=$5
difficult_pose_flag=$6
per_video_flag=$7
nets_repo=$8
experiment_dir=$9
experiments_name=${10}
num_epochs=${11}

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



