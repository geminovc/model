""" Test script to compare prediction quality with H264
    compressed frame as source frame vs. original 
    uncompressed frame
"""
from first_order_model.fom_wrapper import FirstOrderModel
import imageio 
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

frame_update_freq = 10
max_frames = 240

video_name = '/video-conf/scratch/leo_resized.mp4'
dir_name = f'/video-conf/scratch/vibhaa_chunky_directory/leo_logs_ref_every_{frame_update_freq}frames'
video_array = np.array(imageio.mimread(video_name))

original_model = FirstOrderModel("config/api_sample.yaml")
model_with_compressed_vpx_source = FirstOrderModel("config/api_sample.yaml")
psnrs = {'compressed': [], 'original': []}
ssims = {'compressed': [], 'original': []}


# iterate and compute predictions for both source frames on each frame
for i in range(max_frames):
    if i % frame_update_freq == 0:
        original_name = f'{dir_name}/sender_frame_{i:05d}.npy'
        original_source = np.load(original_name)
        original_source_kp = original_model.extract_keypoints(original_source)
        original_model.update_source(original_source, original_source_kp)
        
        compressed_name = f'{dir_name}/reference_frame_{i:05d}.npy'
        compressed_source = np.load(compressed_name)
        compressed_source_kp = model_with_compressed_vpx_source.extract_keypoints(compressed_source)
        model_with_compressed_vpx_source.update_source(compressed_source, compressed_source_kp)
    
    driving = video_array[i, :, :, :] 
    target_kp = original_model.extract_keypoints(driving)
    
    original_prediction = original_model.predict(target_kp)
    psnr = peak_signal_noise_ratio(driving, original_prediction)
    ssim = structural_similarity(driving, original_prediction, multichannel=True)
    psnrs['original'].append(psnr)
    ssims['original'].append(ssim)
    
    compressed_prediction = model_with_compressed_vpx_source.predict(target_kp)
    psnr = peak_signal_noise_ratio(driving, compressed_prediction)
    ssim = structural_similarity(driving, compressed_prediction, multichannel=True)
    psnrs['compressed'].append(psnr)
    ssims['compressed'].append(ssim)


# aggregate metrics
original_psnr_avg = np.mean(psnrs['original'])
original_ssim_avg = np.mean(ssims['original'])
print("Original Frames", original_psnr_avg, original_ssim_avg)

compressed_psnr_avg = np.mean(psnrs['compressed'])
compressed_ssim_avg = np.mean(ssims['compressed'])
print("Compressed Frames", compressed_psnr_avg, compressed_ssim_avg)
