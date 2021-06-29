# Variables from the user
#MAIN_DIR="${HOME}/NETS/nets_implementation/original_bilayer"
MAIN_DIR=../..
machine=${1}
experiment_name=${2}
initialization=${3}
dataset_name=${4}
batch_size=${5}
num_epochs=${6}
test_freq=${7}
metrics_freq=${8}




if [ -z "$dropout_networks" ]
then
    dropout_networks='texture_generator: 0'
fi

# Depending on the machine you run the code from, the directory to save experiments is different due to write access
if [[ "$machine" == "chunky" ]]; then
    experiment_dir=/data/pantea/pantea_experiments_chunky/${dataset_name}/${initialization}
elif [[ "$machine" == "mapmaker" ]]; then
    experiment_dir=/video-conf/scratch/pantea_experiments_mapmaker/${dataset_name}/${initialization}
fi 

# The dataset options for experiment
if [[ "$dataset_name" == "general" ]]; then
    data_root=/video-conf/scratch/pantea/temp_general_extracts
elif [[ "$dataset_name" == "per_person" ]]; then
    data_root=/video-conf/scratch/pantea/random_sampled_per_person
elif [[ "$dataset_name" == "per_video" ]]; then
    data_root=/video-conf/scratch/pantea/temp_per_video_extracts
fi 


# Add initialization options
if [[ "$initialization" == "from_base" ]]; then
    init_networks=' '
    init_experiment_dir='.'
    init_which_epoch='none'

elif [[ "$initialization" == "from_paper" ]]; then
    init_networks='identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator'
    init_experiment_dir=/video-conf/scratch/pantea/bilayer_paper_released/runs/vc2-hq_adrianb_paper_main
    init_which_epoch=2225

fi

cd $MAIN_DIR/

python train.py \
    --experiment_name ${experiment_name} \
    --pretrained_weights_dir /video-conf/scratch/pantea \
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
    --checkpoint_freq 1000 \
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
    --eps 1e-07 \
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
    --lrs 'identity_embedder: 2e-4, texture_generator: 2e-4, keypoints_embedder: 2e-4, inference_generator: 2e-4, discriminator: 2e-4'  \
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
    --visual_freq 100 \
    --wpr_loss_apply_to pred_target_delta_uvs \
    --wpr_loss_decay_schedule '-1' \
    --wpr_loss_type l1 \
    --wpr_loss_weight 0.1 \
    --wpr_loss_weight_decay 1.0 \
    --nme_num_threads 1  \
    --skip_test False \
    --frozen_networks ' ' \
    --unfreeze_texture_generator_last_layers True \
    --unfreeze_inference_generator_last_layers True \
    --replace_AdaSpade_with_conv False \
    --replace_Gtex_output_with_trainable_tensor False \
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
    --use_dropout False \


   
   
