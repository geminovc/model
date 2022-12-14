# Go to the NETS project directory
train_script_dir=$(dirname ${0} 2>&1)
cd $train_script_dir 
cd ../../.. #inside nets_implementation
NETS_DIR=$(pwd)

# Variables from the user
experiment_name=${1}
initialization=${2}
dataset_name=${3}
batch_size=${4}
num_epochs=${5}
test_freq=${6}
metrics_freq=${7}
checkpoint_freq=${8}
visual_freq=${9}
train_dataloader_name=${10}
data_root=${11}
root_to_yaws=${12}
frozen_networks=${13}
unfreeze_texture_generator_last_layers=${14}
unfreeze_inference_generator_last_layers=${15}
experiment_dir=${16}
wpr_loss_weight=${17}
replace_Gtex_output_with_source=${18}
echo "replace_Gtex_output_with_source: $replace_Gtex_output_with_source"
echo "frozen_networks: $frozen_networks"

# Add initialization options
if [[ "$initialization" == "from_base" ]]; then
    init_networks=' '
    init_experiment_dir='.'
    init_which_epoch='none'

elif [[ "$initialization" == "from_personalized" ]]; then
    init_networks='identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator'
    init_experiment_dir=/data/pantea/pantea_experiments_chunky/per_person/from_paper/runs/original_frozen_Gtex_from_identical
    init_which_epoch=2000

elif [[ "$initialization" == "from_bilayer" ]]; then
    init_networks='identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator'
    init_experiment_dir=/video-conf/scratch/pantea/bilayer_paper_released/runs/vc2-hq_adrianb_paper_main
    init_which_epoch=2225

fi
# Find an empty gpu to use


found_empty_gpu=false

while [ "$found_empty_gpu" = false ]
   do
   echo "entering the loop"

   list_of_free_gpus=($(python scripts/gpu_usage.py  2>&1 | tr -d '[],'))
   echo "list of free gpus [${list_of_free_gpus[@]}]"
   
   if ! [ ${#list_of_free_gpus} -eq 0 ]; then
      gpu_number=${list_of_free_gpus[0]}
      echo " gpu ${gpu_number} is empty."
      found_empty_gpu=true
      break
   fi

done

# Based on the conversation with pouya, make the .full file for gpu occupancy on Chunky
host_name=$(hostname)
if [[ "$host_name" == "lab" ]]; then
   filename=/data/pouya/gpu_reserves/${gpu_number}.full
   [[ -f ${filename} ]] || touch ${filename}
fi

cd $NETS_DIR/original_bilayer

CUDA_VISIBLE_DEVICES=${gpu_number} 

python train.py \
--experiment_name ${experiment_name} \
--pretrained_weights_dir /video-conf/scratch/pantea \
--images_log_rate 50 \
--metrics_log_rate 50 \
--random_seed 0 \
--save_dataset_filenames False \
--dataset_load_from_txt False \
--adam_beta1 0.5 \
--adv_loss_weight 0.5 \
--adv_pred_type ragan \
--amp_loss_scale dynamic \
--experiment_dir ${experiment_dir} \
--amp_opt_level  O0 \
--batch_size ${batch_size} \
--bn_momentum 1.0 \
--calc_stats \
--checkpoint_freq ${checkpoint_freq} \
--data_root ${data_root} \
--general_data_root /video-conf/scratch/pantea/temp_general_extracts \
--dis_activation_type leakyrelu \
--dis_downsampling_type avgpool \
--dis_max_channels 512 \
--dis_norm_layer_type bn \
--dis_num_blocks 6 \
--dis_num_channels 64 \
--use_source_background True \
--output_segmentation True \
--dis_output_tensor_size 8 \
--emb_activation_type leakyrelu \
--emb_apply_masks True \
--emb_downsampling_type avgpool \
--emb_max_channels 512 \
--emb_norm_layer_type none \
--emb_num_channels 64 \
--emb_output_tensor_size 8 \
--eps 0.0000001 \
--fem_loss_type l1 \
--fem_loss_weight 10.0 \
--folder_postfix '2d_crop' \
--frame_num_from_paper False \
--inf_activation_type leakyrelu \
--inf_apply_masks True \
--inf_max_channels 256 \
--inf_norm_layer_type ada_bn \
--inf_num_channels 32 \
--inf_pred_segmentation True \
--inf_input_tensor_size 4 \
--inf_pred_source_data False \
--inf_skip_layer_type ada_conv \
--inf_upsampling_type nearest \
--inf_calc_grad True \
--tex_max_channels 512 \
--tex_norm_layer_type ada_spade_bn \
--tex_num_channels 64 \
--tex_pred_segmentation False \
--tex_input_tensor_size 4 \
--tex_skip_layer_type ada_conv \
--tex_upsampling_type nearest \
--tex_activation_type leakyrelu \
--image_size 256 \
--losses_test 'PSNR, lpips, csim, ssim' \
--metrics 'PSNR, lpips, pose_matching, csim, ssim' \
--psnr_loss_apply_to 'pred_target_imgs, target_imgs'  \
--losses_train 'adversarial, feature_matching, perceptual, pixelwise, warping_regularizer, segmentation'  \
--lrs 'identity_embedder: 0.0002, texture_generator: 0.0002, keypoints_embedder: 0.0002, inference_generator: 0.0002, discriminator: 0.0002'  \
--networks_calc_stats 'identity_embedder, texture_generator, keypoints_embedder, inference_generator' \
--networks_test 'identity_embedder, texture_generator, keypoints_embedder, inference_generator' \
--networks_train 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator' \
--num_epochs ${num_epochs} \
--num_gpus 1 \
--num_keypoints 68 \
--num_source_frames 1 \
--num_target_frames 1 \
--num_visuals 1 \
--num_workers_per_process 20 \
--optims 'identity_embedder: adam, texture_generator: adam, keypoints_embedder: adam, inference_generator: adam, discriminator: adam' \
--output_stickmen True \
--per_full_net_names 'vgg19_imagenet_pytorch, vgg16_face_caffe' \
--per_layer_weights '0.03125, 0.0625, 0.125, 0.25, 1.0' \
--per_loss_apply_to 'pred_target_imgs_lf_detached, target_imgs' \
--per_loss_names 'VGG19, VGGFace' \
--per_loss_type 'l1' \
--per_loss_weights '10.0, 0.01' \
--per_net_layers '1,6,11,20,29; 1,6,11,18,25' \
--per_pooling avgpool \
--pix_loss_apply_to 'pred_target_delta_lf_rgbs, target_imgs' \
--pix_loss_names L1 \
--pix_loss_type l1 \
--pix_loss_weights 10.0 \
--project_dir '.' \
--pse_activation_type leakyrelu \
--pse_emb_source_pose False \
--pse_in_channels 136 \
--pse_input_tensor poses \
--pse_num_blocks 4 \
--pse_num_channels 256 \
--runner_name default \
--seg_loss_apply_to 'pred_target_segs_logits, target_segs' \
--seg_loss_names BCE \
--seg_loss_type bce \
--seg_loss_weights 10.0 \
--spn_layers 'conv2d, linear' \
--spn_networks 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator' \
--stats_calc_iters 500 \
--stickmen_thickness 2 \
--test_freq ${test_freq} \
--visual_freq ${visual_freq} \
--wpr_loss_apply_to pred_target_delta_uvs \
--wpr_loss_decay_schedule '-1' \
--wpr_loss_type l1 \
--wpr_loss_weight ${wpr_loss_weight} \
--wpr_loss_weight_decay 1.0 \
--nme_num_threads 1  \
--skip_test False \
--frozen_networks "${frozen_networks}" \
--unfreeze_texture_generator_last_layers ${unfreeze_texture_generator_last_layers} \
--unfreeze_inference_generator_last_layers ${unfreeze_inference_generator_last_layers} \
--replace_AdaSpade_with_conv False \
--replace_Gtex_output_with_trainable_tensor False \
--replace_Gtex_output_with_source ${replace_Gtex_output_with_source} \
--replace_source_specific_with_trainable_tensors False \
--augment_with_general False \
--sample_general_dataset False \
--texture_output_dim 3 \
--use_unet False \
--unet_input_channels 16 \
--unet_output_channels 3 \
--unet_inputs 'lf, hf' \
--metrics_freq ${metrics_freq} \
--metrics_root /video-conf/scratch/pantea/metrics_dataset \
--skip_metrics True \
--init_experiment_dir ${init_experiment_dir} \
--init_networks 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator' \
--init_which_epoch ${init_which_epoch} \
--train_dataloader_name ${train_dataloader_name} \
--visualize_discriminator_scores False \
--root_to_yaws ${root_to_yaws} \

# Based on the conversation with pouya, remove the .full file for gpu occupancy on Chunky
if [[ "$host_name" == "lab" ]]; then
   uname2="$(stat --format '%U' "/data/pouya/gpu_reserves/${gpu_number}.full")"
   if [ "x${uname2}" = "x${USER}" ]; then
      rm /data/pouya/gpu_reserves/${gpu_number}.full
   fi
fi

 