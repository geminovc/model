python  segmentation_generator.py \
    --experiment_name 'test_experiment' \
    --adam_beta1 0.5 \
    --adv_loss_weight 0.5 \
    --adv_pred_type ragan \
    --amp_loss_scale dynamic \
    --amp_opt_level O0 \
    --batch_size 1 \
    --bn_momentum 1.0 \
    --calc_stats \
    --checkpoint_freq 25 \
    --data_root /video-conf/scratch/pantea/Vedantha_dataset \
    --dis_activation_type leakyrelu \
    --dis_downsampling_type avgpool \
    --dis_max_channels 512 \
    --dis_norm_layer_type bn \
    --dis_num_blocks 6 \
    --dis_num_channels 64 \
    --output_segmentation False \
    --dis_output_tensor_size 8 \
    --emb_activation_type leakyrelu \
    --emb_apply_masks False \
    --emb_downsampling_type avgpool \
    --emb_max_channels 512 \
    --emb_norm_layer_type none \
    --emb_num_channels 64 \
    --emb_output_tensor_size 8 \
    --eps 1e-07 \
    --fem_loss_type l1 \
    --fem_loss_weight 10.0 \
    --folder_postfix '2d_crop' \
    --inf_activation_type leakyrelu \
    --inf_apply_masks False \
    --inf_calc_grad False \
    --inf_max_channels 256 \
    --inf_norm_layer_type ada_bn \
    --inf_num_channels 32 \
    --inf_pred_segmentation False \
    --inf_input_tensor_size 4 \
    --inf_pred_source_data False \
    --inf_skip_layer_type ada_conv \
    --inf_upsampling_type nearest \
    --tex_max_channels 512 \
    --tex_norm_layer_type ada_spade_bn \
    --tex_num_channels 64 \
    --tex_pred_segmentation False \
    --tex_input_tensor_size 4 \
    --tex_skip_layer_type ada_conv \
    --tex_upsampling_type nearest \
    --tex_activation_type leakyrelu \
    --image_size 256 \
    --label_run : name \
    --losses_test 'lpips, csim' \
    --losses_train 'adversarial, feature_matching, perceptual, pixelwise, warping_regularizer'  \
    --lrs 'identity_embedder: 2e-4, texture_generator: 2e-4, keypoints_embedder: 2e-4, inference_generator: 2e-4, discriminator: 2e-4'  \
    --networks_calc_stats 'identity_embedder, texture_generator, keypoints_embedder, inference_generator' \
    --networks_test 'identity_embedder, texture_generator, keypoints_embedder, inference_generator' \
    --networks_to_train 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator' \
    --networks_train 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator' \
    --num_epochs 4000 \
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
    --project_dir '/data/pantea' \
    --pse_activation_type leakyrelu \
    --pse_emb_source_pose False \
    --pse_in_channels 136 \
    --pse_input_tensor poses \
    --pse_num_blocks 4 \
    --pse_num_channels 256 \
    --pse_use_harmonic_enc False \
    --random_seed 0 \
    --runner_name default \
    --seg_loss_apply_to 'pred_target_segs_logits, target_segs' \
    --seg_loss_names BCE \
    --seg_loss_type bce \
    --seg_loss_weights 10.0 \
    --spn_layers 'conv2d, linear' \
    --spn_networks 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator' \
    --stats_calc_iters 500 \
    --stickmen_thickness 2 \
    --test_freq 5 \
    --visual_freq '-1' \
    --wpr_loss_apply_to pred_target_delta_uvs \
    --wpr_loss_decay_schedule '-1' \
    --wpr_loss_type l1 \
    --wpr_loss_weight 0.0 \
    --wpr_loss_weight_decay 1.0  
   