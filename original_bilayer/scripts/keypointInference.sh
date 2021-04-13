cd ../
python  keypoint_segmentation_generator.py \
    --experiment_name 'test_experiment' \
    --adam_beta1 0.5 \
    --adv_loss_weight 0.5 \
    --adv_pred_type ragan \
