# The input values
phase=${1}
video_root=${2}
sampling_rate=${3}
data_root=${4}

python  extract.py \
    --pretrained_weights_dir /video-conf/scratch/pantea \
    --phase ${phase} \
    --video_root ${video_root} \
    --sampling_rate ${sampling_rate} \
    --experiment_name 'extraction' \
    --batch_size 10000 \
    --data_root ${data_root} \
    --output_segmentation True \
    --image_size 256 \
    --num_gpus 1 \
    --num_keypoints 68 \
    --num_source_frames 1 \
    --num_target_frames 1 \
    --num_visuals 1 \
    --num_workers_per_process 20 \
    --output_stickmen False \
    --project_dir '../' \
    --random_seed 0 \
