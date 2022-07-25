import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
from sync_batchnorm import DataParallelWithCallback
from skimage.metrics import peak_signal_noise_ratio
from skimage import img_as_float32
from skimage.transform import resize
from skimage.metrics import structural_similarity
from frames_dataset import get_num_frames, get_frame
from modules.model import Vgg19
import piq
import subprocess
import av

import matplotlib
import matplotlib.pyplot as plt

reference_frame_list = []

from aiortc.codecs.vpx import Vp8Encoder, Vp8Decoder, vp8_depayload
from aiortc.jitterbuffer import JitterFrame
KEYPOINT_FIXED_PAYLOAD_SIZE = 125 # bytes


def get_size_of_nested_list(list_of_elem):
    """ helper to get size of nested parameter list """ 
    count = 0
    for elem in list_of_elem:
        if type(elem) == list:  
            count += get_size_of_nested_list(elem)
        else:
            count += 1    
    return count


def get_model_info(log_dir, kp_detector, generator):
    """ get model summary information for the passed in keypoint detector and 
        generator in a text file in the log directory """
    with open(os.path.join(log_dir, 'model_summary.txt'), 'wt') as model_file:
        for model, name in zip([kp_detector, generator], ['kp', 'generator']):
            number_of_trainable_parameters = 0
            total_number_of_parameters = 0
            if model is not None:
                for param in model.parameters():
                    total_number_of_parameters += get_size_of_nested_list(list(param))
                    if param.requires_grad:
                        number_of_trainable_parameters += get_size_of_nested_list(list(param))

                model_file.write('%s %s: %s\n' % (name, 'total_number_of_parameters',
                        str(total_number_of_parameters)))
                model_file.write('%s %s: %s\n' % (name, 'number_of_trainable_parameters',
                        str(number_of_trainable_parameters)))

def get_avg_visual_metrics(visual_metrics):
    """ get average of visual metrics across all frames """
    psnrs = [m['psnr'] for m in visual_metrics]
    ssims = [m['ssim'] for m in visual_metrics]
    lpips_list = [m['lpips'] for m in visual_metrics]
    return np.mean(psnrs), np.mean(ssims), np.mean(lpips_list)


def frame_to_tensor(frame, device):
    """ convert numpy arrays to tensors for reconstruction pipeline """
    array = np.expand_dims(frame, 0).transpose(0, 3, 1, 2)
    array = torch.from_numpy(array)
    return array.float().to(device)

ssim_correlation_file = open('ssim_data_threshold_approach.txt', 'w+')
ssim_correlation_file.write('video,frame,dist,ssim\n')
def nearness_check(ref_frame, ref_kp, frame, kp_frame, method='single_reference', video_num=1, frame_idx=0):
    """ checks if two frames are close enough to not be treated as new reference """
    global ssim_correlation_file

    threshold = float('inf')
    if method == 'single_reference':
        return True

    elif method == 'kp_l2_distance':
        xy_frame = kp_frame['value']
        xy_ref = ref_kp['value']
        dist = torch.dist(xy_frame, xy_ref, p=2)
        ssim_correlation_file.write(f'{video_num},{frame_idx},{dist.data.cpu().numpy():.4f},')
        if dist > threshold:
            return False

    elif method == 'ssim':
        dist = piq.ssim(ref_frame, frame, data_range=1.).data.cpu().numpy().flatten()[0]
        ssim_correlation_file.write(f'{video_num},{frame_idx},{dist:.4f},')
        if dist > threshold:
            return False
    return True


def find_best_reference_frame(cur_frame, cur_kp, video_num=1, frame_idx=0):
    """ find the best reference frame for this current frame """
    global reference_frame_list
    for (ref_s, ref_kp) in reversed(reference_frame_list):
        if nearness_check(ref_s, ref_kp, cur_frame, cur_kp, \
                method='ssim', video_num=video_num, frame_idx=frame_idx):
            return True, ref_s, ref_kp

    #reference_frame_list.append((cur_frame, cur_kp))
    return False, cur_frame, cur_kp


def get_frame_from_video_codec(av_frame, encoder, decoder, quantizer):
    """ go through the encoder/decoder pipeline to get a 
        representative decoded frame
    """
    payloads, timestamp = encoder.encode(av_frame, quantizer=quantizer)
    payload_data = [vp8_depayload(p) for p in payloads]
    jitter_frame = JitterFrame(data=b"".join(payload_data), timestamp=timestamp)
    decoded_frames = decoder.decode(jitter_frame)
    decoded_frame = decoded_frames[0].to_rgb().to_ndarray()
    return decoded_frame, sum([len(p) for p in payloads])


def get_bitrate(stream, video_duration):
    """ get bitrate (in kbps) from a sequence of frame sizes and video
        duration in seconds
    """
    total_bytes = np.sum(stream)
    return total_bytes * 8 / video_duration / 1000.0


def get_video_duration(filename):
    """ get duration of video in seconds 
    """
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)


def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset, timing_enabled, 
        save_visualizations_as_images, experiment_name, reference_frame_update_freq=None):
    """ reconstruct driving frames for each video in the dataset using the first frame
        as a source frame. Config specifies configuration details, while timing 
        determines whether to time the functions on a gpu or not """
    global ssim_correlation_file
    global reference_frame_list
    log_dir = os.path.join(log_dir, 'reconstruction' + '_' + experiment_name)
    png_dir = os.path.join(log_dir, 'png')
    visualization_dir = os.path.join(log_dir, 'visualization')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator_params = config['model_params']['generator_params']
    generator_type = generator_params.get('generator_type', 'occlusion_aware')
    
    choose_reference_frame = False
    use_same_tgt_ref_quality = False
    
    if generator_type not in ['vpx', 'bicubic']:
        if checkpoint is not None:
            dense_motion = generator.dense_motion_network if generator_type == 'occlusion_aware' else None
            Logger.load_cpk(checkpoint, generator=generator, 
                    kp_detector=kp_detector, device=device, 
                    dense_motion_network=dense_motion, generator_type=generator_type, reconstruction=True)
        else:
            raise AttributeError('Checkpoint should be specified for reconstruction')
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
    
    
    metrics_file = open(os.path.join(log_dir, experiment_name + '_metrics_summary.txt'), 'wt')
    frame_metrics_file = open(os.path.join(log_dir, experiment_name + '_per_frame_metrics.txt'), 'wt')
    loss_list = []
    vgg_model = Vgg19()
    if torch.cuda.is_available():
        if generator is not None:
            generator = DataParallelWithCallback(generator)
        if kp_detector is not None:
            kp_detector = DataParallelWithCallback(kp_detector)
        vgg_model = vgg_model.cuda()
 
    loss_fn_vgg = vgg_model.compute_loss
    if generator is not None:
        generator.eval()
    if kp_detector is not None:
        kp_detector.eval()

    # get number of model parameters and timing stats
    get_model_info(log_dir, kp_detector, generator)
    start = torch.cuda.Event(enable_timing=timing_enabled)
    end = torch.cuda.Event(enable_timing=timing_enabled)
    for it, x in tqdm(enumerate(dataloader)):
        updated_src = 0
        reference_stream = []
        lr_stream = []
        
        hr_encoder, lr_encoder = Vp8Encoder(), Vp8Encoder()
        hr_decoder, lr_decoder = Vp8Decoder(), Vp8Decoder()
        
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break

        with torch.no_grad():
            predictions = []
            visualizations = []
            visual_metrics = []
            video_name = x['video_path'][0]
            print('doing video', video_name)
            video_duration = get_video_duration(video_name)

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            if timing_enabled:
                source_time = start.elapsed_time(end)
                driving_times, generator_times, visualization_times = [], [], []
            
            container = av.open(file=video_name, format=None, mode='r')
            stream = container.streams.video[0]

            frame_idx = 0
            for av_frame in container.decode(stream):
                # ground-truth
                frame = av_frame.to_rgb().to_ndarray()
                driving = frame_to_tensor(img_as_float32(frame), device)
                
                # get LR video frame
                driving_lr = F.interpolate(driving, 64).data.cpu().numpy()
                driving_lr = np.transpose(driving_lr, [0, 2, 3, 1])[0]
                driving_lr *= 255
                driving_lr = driving_lr.astype(np.uint8)
                
                driving_lr_av = av.VideoFrame.from_ndarray(driving_lr)
                driving_lr_av.pts = av_frame.pts
                driving_lr_av.time_base = av_frame.time_base
                
                driving_lr, compressed_tgt = get_frame_from_video_codec(driving_lr_av, lr_encoder, lr_decoder, 5)
                driving_lr = frame_to_tensor(img_as_float32(driving_lr), device)
                
                # for use as source frame
                decoded_frame, compressed_src = get_frame_from_video_codec(av_frame, hr_encoder, hr_decoder, 32)
                decoded_frame = img_as_float32(decoded_frame)
                decoded_tensor = frame_to_tensor(decoded_frame, device)
                update_source = False if not use_same_tgt_ref_quality else True
                
                if kp_detector is not None:
                    if frame_idx == 0: 
                        source = decoded_tensor
                        start.record()
                        kp_source = kp_detector(source)
                        end.record()
                        torch.cuda.synchronize()
                        update_source = True
                        reference_frame_list.append((source, kp_source))
                        reference_stream.append(compressed_src)

                    if reference_frame_update_freq is not None:
                        if frame_idx % reference_frame_update_freq == 0:
                            source = decoded_tensor
                            kp_source = kp_detector(source)
                            update_source = True
                            reference_stream.append(compressed_src)

                    elif choose_reference_frame:
                        # runs at sender, so use frame prior to encode/decode pipeline
                        cur_frame = frame_to_tensor(frame, device) 
                        cur_kp = kp_detector(cur_frame)

                        frame_reuse, source, kp_source = find_best_reference_frame(cur_frame, cur_kp, \
                            video_num=it+1, frame_idx=frame_idx)
                else:
                    # default if there's no KP based method
                    source = decoded_tensor

                frame_idx += 1
                if generator_params.get('use_lr_video', False):
                    lr_stream.append(compressed_tgt)
                else:
                    lr_stream.append(KEYPOINT_FIXED_PAYLOAD_SIZE)
                
                if kp_detector is not None:
                    start.record()
                    if generator_params.get('use_lr_video', False):
                        kp_driving = kp_detector(driving_lr)
                    else:
                        kp_driving = kp_detector(driving)
                    end.record()
                    torch.cuda.synchronize()
                    if timing_enabled:
                        driving_times.append(start.elapsed_time(end))
                
                start.record()
                if generator_type in ['occlusion_aware', 'split_hf_lf']:
                    out = generator(source, kp_source=kp_source, \
                            kp_driving=kp_driving, update_source=update_source, driving_lr=driving_lr)

                    if use_same_tgt_ref_quality:
                        ref_out = generator(driving, kp_source=kp_driving, \
                            kp_driving=kp_driving, update_source=True, driving_lr=driving_lr)

                elif generator_type == "bicubic":
                    out = {'prediction': F.interpolate(driving_lr, source.shape[2], mode='bicubic')}
                    lr_stream.append(compressed_tgt)
                
                elif generator_type == "vpx":
                    out = {'prediction': decoded_tensor}
                    reference_stream.append(compressed_src)

                else:
                    out = generator(driving_lr)
                    lr_stream.append(compressed_tgt)
                
                end.record()
                torch.cuda.synchronize()
                if timing_enabled:
                    generator_times.append(start.elapsed_time(end))

                out['prediction'] = torch.clamp(out['prediction'], min=0, max=1)
                
                ssim = piq.ssim(driving, out['prediction'], data_range=1.).data.cpu().numpy().flatten()[0]
                if use_same_tgt_ref_quality:
                    ref_ssim = piq.ssim(driving, ref_out['prediction'], data_range=1.).data.cpu().numpy().flatten()[0]
                    if ssim < ref_ssim:
                        source = driving
                        kp_source = kp_driving
                        out = ref_out
                        reference_frame_list = [(source, kp_source)]
                        updated_src += 1
                        reference_stream.append(compressed_src)
                        ssim = ref_ssim
                
                if choose_reference_frame:
                    ssim_correlation_file.write(f'{ssim:.4f}\n')
                else:
                    ssim_correlation_file.write(f'{it+1},{frame_idx},{ssim:.4f}\n')
                                
                if kp_detector is not None:
                    out['kp_source'] = kp_source
                    out['kp_driving'] = kp_driving
                    del out['sparse_deformed']
                
                start.record()
                visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
                                                                                    driving=driving, out=out)
                end.record()
                torch.cuda.synchronize()
                if timing_enabled:
                    visualization_times.append(start.elapsed_time(end))
                visualizations.append(visualization)

                if frame_idx % 50 == 0:
                    #print(f'finished {frame_idx} frames')
                    if save_visualizations_as_images:
                        for i, v in enumerate(visualizations):
                            frame_name = x['name'][0] + '_frame' + str(frame_idx - 50 + i) + '.png'
                            imageio.imsave(os.path.join(visualization_dir, frame_name), v)
                    image_name = f"{x['name'][0]}_{frame_idx}_{config['reconstruction_params']['format']}"
                    print(f'saving {frame_idx} frames: updated src {updated_src}')
                    imageio.mimsave(os.path.join(log_dir, image_name), visualizations)
                    visualizations = []

                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())
                visual_metrics.append(Logger.get_visual_metrics(out['prediction'], driving, loss_fn_vgg))
                
            container.close()
            print('total frames', frame_idx, 'updated src', updated_src)
            ref_br = get_bitrate(reference_stream, video_duration)
            lr_br = get_bitrate(lr_stream, video_duration)
            
            psnr, ssim, lpips_val = get_avg_visual_metrics(visual_metrics)
            metrics_file.write(f'{x["name"][0]} PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips_val}, ' +
                    f'Reference: {ref_br:.3f}Kbps, LR: {lr_br:.3f}Kbps \n')
            metrics_file.flush()

            if it == 0:
                frame_metrics_file.write('video_num,frame,psnr,ssim,lpips\n')
            for i, m in enumerate(visual_metrics):
                frame_metrics_file.write(f'{it + 1},{i},{m["psnr"][0]},{m["ssim"]},{m["lpips"]}\n')
            frame_metrics_file.flush()

            if save_visualizations_as_images:
                for i, v in enumerate(visualizations):
                    frame_name = x['name'][0] + '_frame' + str(frame_idx - len(visualizations) + i) + '.png'
                    imageio.imsave(os.path.join(visualization_dir, frame_name), v)
            image_name = f"{x['name'][0]}_{frame_idx}_{config['reconstruction_params']['format']}"
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)
            visualizations = []
            
            if timing_enabled:
                print('source keypoints:', source_time, 'driving:', np.average(driving_times), \
                    'generator:', np.average(generator_times),'visualization:', np.average(visualization_times))

    print('Reconstruction loss: %s' % np.mean(loss_list))
    metrics_file.write('Reconstruction loss: %s\n' % np.mean(loss_list))
    metrics_file.close()
    frame_metrics_file.close()
    ssim_correlation_file.close()

