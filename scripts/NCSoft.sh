
# Do the per_video voxceleb2 train dataloader
# data_root=/video-conf/scratch/pantea/per_video_NCSoft
# root_to_yaws=/video-conf/scratch/pantea/per_video_NCSoft_yaws/angles
video_id=fd_mtL88o1k 
# cd /home/pantea/NETS/fix_prs/nets_implementation/original_bilayer/scripts/automated

#CUDA_VISIBLE_DEVICES=2 ./train_script.sh  "chunky" per_video_from_my_model_voxceleb_fd_mtL88o1k  "from_paper" "per_video" 2 30 1 1 5 5 'voxceleb2' ${data_root} ${root_to_yaws}


# At this point you have all the models, make the images with the models

make_images () {
cd /home/pantea/NETS/video_trials/nets_implementation/original_bilayer/examples

# /video-conf/scratch/pantea_experiments_mapmaker/per_person/from_paper/runs/more_yaws_baseline_voxceleb2_easy_diff_combo

python infer_test.py \
--experiment_dir '/video-conf/scratch/pantea_experiments_mapmaker/per_person/from_paper' \
--experiment_name 'more_yaws_baseline_voxceleb2_easy_diff_combo' \
--which_epoch 2000 \
--dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
--source_relative_path ${source_relative_path} \
--target_relative_path ${target_relative_path} \
--save_dir ${save_dir}/per_person_voxceleb2_${video_id} \


# python infer_test.py \
# --experiment_dir '/video-conf/scratch/pantea/bilayer_paper_released' \
# --experiment_name 'vc2-hq_adrianb_paper_main' \
# --which_epoch 2225 \
# --dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
# --source_relative_path ${source_relative_path} \
# --target_relative_path ${target_relative_path} \
# --save_dir ${save_dir}/bilayer_${video_id} \



# python infer_test.py \
# --experiment_dir '/data/pantea/pantea_experiments_chunky/per_person/from_paper' \
# --experiment_name 'original_frozen_Gtex_from_identical' \
# --which_epoch 3000 \
# --dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
# --source_relative_path ${source_relative_path} \
# --target_relative_path ${target_relative_path} \
# --save_dir ${save_dir}/per_person_yaw_${video_id} \

# ### Append the images in a single strip

# ffmpeg -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
# -i ${save_dir}/bilayer_${video_id}/pred_target_False_False.png \
# -i ${save_dir}/per_person_voxceleb2_${video_id}/pred_target_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/pred_target_False_False.png \
# -filter_complex "[0][1][2][3][4]hstack=inputs=5" ${save_dir}/stacked_per_person_with_yaw.png

# ffmpeg -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
# -i ${save_dir}/bilayer_${video_id}/pred_target_False_False.png \
# -i ${save_dir}/per_person_voxceleb2_${video_id}/pred_target_False_False.png \
# -filter_complex "[0][1][2][3]hstack=inputs=4" ${save_dir}/stacked_per_person_no_yaw.png

# if  [ "$difficult_pose_flag" = true ]
# then 


# ffmpeg -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
# -i ${save_dir}/bilayer_${video_id}/pred_tex_hf_rgbs_False_False.png \
# -i ${save_dir}/bilayer_${video_id}/pred_target_False_False.png \
# -filter_complex "[0][1][2][3]hstack=inputs=4" ${save_dir}/bilayer_with_pred_tex.png

# ffmpeg -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
# -i ${save_dir}/bilayer_${video_id}/pred_hf_False_False.png \
# -i ${save_dir}/bilayer_${video_id}/pred_target_False_False.png \
# -filter_complex "[0][1][2][3]hstack=inputs=4" ${save_dir}/bilayer_with_pred_hf.png


# ffmpeg -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
# -i ${save_dir}/per_person_voxceleb2_${video_id}/pred_tex_hf_rgbs_False_False.png \
# -i ${save_dir}/per_person_voxceleb2_${video_id}/pred_target_False_False.png \
# -filter_complex "[0][1][2][3]hstack=inputs=4" ${save_dir}/per_person_voxceleb2_${video_id}_with_pred_tex.png

# ffmpeg -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
# -i ${save_dir}/per_person_voxceleb2_${video_id}/pred_hf_False_False.png \
# -i ${save_dir}/per_person_voxceleb2_${video_id}/pred_target_False_False.png \
# -filter_complex "[0][1][2][3]hstack=inputs=4" ${save_dir}/per_person_voxceleb2_${video_id}_with_pred_hf.png

# ffmpeg -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/pred_tex_hf_rgbs_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/pred_target_False_False.png \
# -filter_complex "[0][1][2][3]hstack=inputs=4" ${save_dir}/per_person_yaw_${video_id}_with_pred_tex.png

# ffmpeg -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/pred_hf_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/pred_target_False_False.png \
# -filter_complex "[0][1][2][3]hstack=inputs=4" ${save_dir}/per_person_yaw_${video_id}_with_pred_hf.png

# ffmpeg -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
# -i ${save_dir}/per_person_voxceleb2_${video_id}/pred_target_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/pred_target_False_False.png \
# -filter_complex "[0][1][2][3]hstack=inputs=4" ${save_dir}/close_random_compare.png


# fi

# if  [ "$per_video_flag" = true ]
# then 

# python infer_test.py \
# --experiment_dir '/data/pantea/pantea_experiments_chunky/per_video/from_paper' \
# --experiment_name per_video_from_my_model_voxceleb_${video_id} \
# --which_epoch 30 \
# --dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
# --source_relative_path ${source_relative_path} \
# --target_relative_path ${target_relative_path} \
# --save_dir ${save_dir}/per_video_voxceleb2_${video_id} \

# python infer_test.py \
# --experiment_dir '/data/pantea/pantea_experiments_chunky/per_video/from_paper' \
# --experiment_name per_video_from_my_model_${video_id} \
# --which_epoch 30 \
# --dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
# --source_relative_path ${source_relative_path} \
# --target_relative_path ${target_relative_path} \
# --save_dir ${save_dir}/per_video_yaw_${video_id} \

# ffmpeg -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
# -i ${save_dir}/bilayer_${video_id}/pred_target_False_False.png \
# -i ${save_dir}/per_person_voxceleb2_${video_id}/pred_target_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/pred_target_False_False.png \
# -i ${save_dir}/per_video_voxceleb2_${video_id}/pred_target_False_False.png \
# -i ${save_dir}/per_video_yaw_${video_id}/pred_target_False_False.png \
# -filter_complex "[0][1][2][3][4][5][6]hstack=inputs=7" ${save_dir}/stacked_per_video_with_yaw.png

# ffmpeg -i ${save_dir}/per_person_yaw_${video_id}/masked_source_imgs_False_False.png \
# -i ${save_dir}/per_person_yaw_${video_id}/masked_target_imgs_False_False.png \
# -i ${save_dir}/bilayer_${video_id}/pred_target_False_False.png \
# -i ${save_dir}/per_person_voxceleb2_${video_id}/pred_target_False_False.png \
# -i ${save_dir}/per_video_voxceleb2_${video_id}/pred_target_False_False.png \
# -filter_complex "[0][1][2][3][4]hstack=inputs=5" ${save_dir}/stacked_per_video_no_yaw.png

# fi


}

difficult_pose_flag=false
# ## test
# per_video_flag=false
# save_dir=/data/pantea/NCSoft/images/test
# source_relative_path=test/id00015/0fijmz4vTVU/00002/0   
# target_relative_path=test/id00015/0fijmz4vTVU/00002/10 
# make_images


## unseen_test
per_video_flag=true
video_id=fd_mtL88o1k 
save_dir=/data/pantea/NCSoft/images/third_verion/unseen_test
source_relative_path=unseen_test/id00015/fd_mtL88o1k/00344/0 
target_relative_path=unseen_test/id00015/fd_mtL88o1k/00344/10 
make_images


# difficult_pose_flag=true
# ## test on difficult pose
# per_video_flag=false
# video_id=XvqWxjZySuA
# save_dir=/data/pantea/NCSoft/images/test_difficult_pose_new_person_no_frozen
# source_relative_path=test/id00015/XvqWxjZySuA/00278/0 
# target_relative_path=test/id00015/XvqWxjZySuA/00278/1 
# make_images

# video_id=0iQWqFw6FOU
# per_video_flag=false
# save_dir=/data/pantea/NCSoft/images/test_2_difficult_pose_new_person_no_frozen
# source_relative_path=test/id00015/0iQWqFw6FOU/00014/0 
# target_relative_path=test/id00015/0iQWqFw6FOU/00014/10  
# make_images

# per_video_flag=false
# video_id=Jf2SpxRNlrk
# save_dir=/data/pantea/NCSoft/images/unseen_test_difficult_pose_new_person_no_frozen
# source_relative_path=unseen_test/id00015/Jf2SpxRNlrk/00189/0 
# target_relative_path=unseen_test/id00015/Jf2SpxRNlrk/00189/1 
# make_images


make_videos () {
cd /home/pantea/NETS/video_trials/nets_implementation/original_bilayer/examples

for video_technique in representative_sources last_frame_next_source last_predicted_next_source
do

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

python infer_video.py \
--experiment_dir '/data/pantea/pantea_experiments_chunky/per_video/from_paper' \
--experiment_name per_video_from_my_model_${video_id} \
--which_epoch 30 \
--video_path /video-conf/vedantha/voxceleb2/dev/mp4/id00015/${video_id}/${video_path} \
--dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
--relative_path_base ${relative_path_base}  \
--yaw_root '/video-conf/scratch/pantea/pose_results/yaws/per_person_1_three_datasets/angles' \
--save_dir ${save_dir}/per_video_yaw_${video_id} \
--video_technique ${video_technique}

python infer_video.py \
--experiment_dir '/data/pantea/pantea_experiments_chunky/per_video/from_paper' \
--experiment_name per_video_from_my_model_voxceleb_${video_id} \
--which_epoch 30 \
--video_path /video-conf/vedantha/voxceleb2/dev/mp4/id00015/${video_id}/${video_path} \
--dataset_root '/video-conf/scratch/pantea/per_person_1_three_datasets' \
--relative_path_base ${relative_path_base}  \
--yaw_root '/video-conf/scratch/pantea/pose_results/yaws/per_person_1_three_datasets/angles' \
--save_dir ${save_dir}/per_video_voxceleb2_${video_id} \
--video_technique ${video_technique}


done

ffmpeg -i ${save_dir}/per_video_yaw_${video_id}/masked_original.mp4 \
-i ${save_dir}/bilayer_${video_id}/representative_sources.mp4 \
-i ${save_dir}/per_video_yaw_${video_id}/representative_sources.mp4 \
-filter_complex "[0][1][2]hstack=inputs=3" ${save_dir}/stacked_representative_sources_orig_by_ours_with_yaw.mp4


ffmpeg -i ${save_dir}/per_video_yaw_${video_id}/masked_original.mp4 \
-i ${save_dir}/bilayer_${video_id}/last_frame_next_source.mp4 \
-i ${save_dir}/per_video_yaw_${video_id}/last_frame_next_source.mp4 \
-filter_complex "[0][1][2]hstack=inputs=3" ${save_dir}/stacked_last_frame_next_source_orig_by_ours_with_yaw.mp4


ffmpeg -i ${save_dir}/per_video_yaw_${video_id}/masked_original.mp4 \
-i ${save_dir}/bilayer_${video_id}/last_frame_next_source.mp4 \
-i ${save_dir}/per_person_yaw_${video_id}/last_frame_next_source.mp4 \
-i ${save_dir}/per_video_yaw_${video_id}/last_frame_next_source.mp4 \
-filter_complex "[0][1][2][3]hstack=inputs=4" ${save_dir}/stacked_last_frame_next_source_with_yaw.mp4


ffmpeg -i ${save_dir}/per_video_yaw_${video_id}/masked_original.mp4 \
-i ${save_dir}/bilayer_${video_id}/representative_sources.mp4 \
-i ${save_dir}/per_person_yaw_${video_id}/representative_sources.mp4 \
-i ${save_dir}/per_video_yaw_${video_id}/representative_sources.mp4 \
-filter_complex "[0][1][2][3]hstack=inputs=4" ${save_dir}/stacked_representative_sources_with_yaw.mp4


ffmpeg -i ${save_dir}/per_video_yaw_${video_id}/masked_original.mp4 \
-i ${save_dir}/bilayer_${video_id}/last_frame_next_source.mp4 \
-i ${save_dir}/per_person_voxceleb2_${video_id}/last_frame_next_source.mp4 \
-i ${save_dir}/per_person_yaw_${video_id}/last_frame_next_source.mp4 \
-i ${save_dir}/per_video_voxceleb2_${video_id}/last_frame_next_source.mp4 \
-i ${save_dir}/per_video_yaw_${video_id}/last_frame_next_source.mp4 \
-filter_complex "[0][1][2][3][4][5]hstack=inputs=6" ${save_dir}/stacked_last_frame_next_source_all.mp4


ffmpeg -i ${save_dir}/per_video_yaw_${video_id}/masked_original.mp4 \
-i ${save_dir}/bilayer_${video_id}/representative_sources.mp4 \
-i ${save_dir}/per_person_voxceleb2_${video_id}/representative_sources.mp4 \
-i ${save_dir}/per_person_yaw_${video_id}/representative_sources.mp4 \
-i ${save_dir}/per_video_voxceleb2_${video_id}/representative_sources.mp4 \
-i ${save_dir}/per_video_yaw_${video_id}/representative_sources.mp4 \
-filter_complex "[0][1][2][3][4][5]hstack=inputs=6" ${save_dir}/stacked_representative_sources_all.mp4


}


## unseen_test
per_video_flag=true
video_id=fd_mtL88o1k 
video_path=$(ls /video-conf/vedantha/voxceleb2/dev/mp4/id00015/${video_id} | sort -V | tail -n 1)
save_dir=/data/pantea/NCSoft/videos/third_verion/unseen_test
relative_path_base=unseen_test/id00015/fd_mtL88o1k/00344
# make_videos


# per_video_flag=true
# video_id=Jf2SpxRNlrk 
# video_path=$(ls /video-conf/vedantha/voxceleb2/dev/mp4/id00015/${video_id} | sort -V | tail -n 1)
# save_dir=/data/pantea/NCSoft/videos/unseen_test_difficult_pose_4
# relative_path_base=unseen_test/id00015/Jf2SpxRNlrk/00209
# # make_videos