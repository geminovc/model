import os
from tqdm import tqdm
import torch
from  shrink_util import set_gen_module, set_keypoint_module
import torch.nn.functional as F
from torch.utils.data import DataLoader
from first_order_model.logger import Logger, Visualizer
import numpy as np
import imageio
from first_order_model.sync_batchnorm import DataParallelWithCallback
from skimage.metrics import peak_signal_noise_ratio
from skimage import img_as_float32
from skimage.transform import resize
from skimage.metrics import structural_similarity
from first_order_model.frames_dataset import get_num_frames, get_frame
from first_order_model.modules.model import Vgg19, VggFace16
from first_order_model.utils import frame_to_tensor, get_model_macs, get_model_info
import piq
import subprocess
import av
import lpips

import matplotlib
import matplotlib.pyplot as plt

reference_frame_list = []
encode_using_vpx = False

from aiortc.codecs.vpx import Vp8Encoder, Vp8Decoder, Vp9Encoder, Vp9Decoder, vp8_depayload
from aiortc.jitterbuffer import JitterFrame
KEYPOINT_FIXED_PAYLOAD_SIZE = 125 # bytes
special_frames_list = [1322, 574, 140, 1786, 1048, 839, 761, 2253, 637, 375, \
        1155, 2309, 1524, 1486, 1207, 315, 1952, 2111, 2148, 1530, \
        112, 939, 1211, 403, 2225, 1900, 207, 1634, 2006, 28]  
SAVE_LR_FRAMES = True
generate_video_visualizations = False


def get_avg_visual_metrics(visual_metrics):
    """ get average of visual metrics across all frames """
    psnrs = [m['psnr'] for m in visual_metrics]
    ssims = [m['ssim'] for m in visual_metrics]
    ssim_dbs = [m['ssim_db'] for m in visual_metrics]
    lpips_list = [m['lpips'] for m in visual_metrics]
    orig_lpips_list = [m['orig_lpips'] for m in visual_metrics]
    face_lpips_list = [m['face_lpips'] for m in visual_metrics]
    return np.mean(psnrs), np.mean(ssims), np.mean(lpips_list), np.mean(ssim_dbs), \
            np.mean(orig_lpips_list), np.mean(face_lpips_list)


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


def get_frame_from_video_codec(av_frame, av_frame_index, encoder, decoder, quantizer=-1, bitrate=None):
    """ go through the encoder/decoder pipeline to get a 
        representative decoded frame
    """
    # stamp the frame
    #av_frame = stamp_frame(av_frame, av_frame_index, av_frame.pts, av_frame.time_base)

    if bitrate == None:
        payloads, timestamp = encoder.encode(av_frame, quantizer=quantizer, enable_gcc=False)
    else:
        payloads, timestamp = encoder.encode(av_frame, quantizer=-1, \
                target_bitrate=bitrate, enable_gcc=False)
    payload_data = [vp8_depayload(p) for p in payloads]
    jitter_frame = JitterFrame(data=b''.join(payload_data), timestamp=timestamp)
    decoded_frames = decoder.decode(jitter_frame)
    decoded_frame_av = decoded_frames[0]
    #decoded_frame_av, video_frame_index = destamp_frame(decoded_frames[0])
    decoded_frame = decoded_frame_av.to_rgb().to_ndarray()
    return decoded_frame_av, decoded_frame, sum([len(p) for p in payloads])


def get_bitrate(stream, video_duration):
    """ get bitrate (in kbps) from a sequence of frame sizes and video
        duration in seconds
    """
    total_bytes = np.sum(stream)
    return total_bytes * 8 / video_duration / 1000.0


def get_video_duration(filename):
    """ get duration of video in seconds 
    """
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                             'format=duration', '-of',
                             'default=noprint_wrappers=1:nokey=1', filename],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)


def resize_tensor_to_array(input_tensor, output_size, device, mode='nearest'):
    """ resizes a float tensor of range 0.0-1.0 to an int numpy array
        of output_size
    """
    output_array = F.interpolate(input_tensor, output_size, mode=mode).data.cpu().numpy()
    output_array = np.transpose(output_array, [0, 2, 3, 1])[0]
    output_array *= 255
    output_array = output_array.astype(np.uint8)
    return output_array


def write_in_file(input_file, info):
    input_file.write(info)
    input_file.flush()

NUM_ROWS = 10
NUMBER_OF_BITS = 16

def stamp_frame(frame, frame_index, frame_pts, frame_time_base):
    """ stamp frame with barcode for frame index before transmission
    """
    frame_array = frame.to_rgb().to_ndarray()
    stamped_frame = np.zeros((frame_array.shape[0] + NUM_ROWS,
                            frame_array.shape[1], frame_array.shape[2]))
    k = frame_array.shape[1] // NUMBER_OF_BITS
    stamped_frame[:-NUM_ROWS, :, :] = frame_array
    id_str = f'{frame_index+1:0{NUMBER_OF_BITS}b}'

    for i in range(len(id_str)):
        if id_str[i] == '0':
            for j in range(k):
                for s in range(NUM_ROWS):
                    stamped_frame[-s-1, i * k + j, 0] = 0
                    stamped_frame[-s-1, i * k + j, 1] = 0
                    stamped_frame[-s-1, i * k + j, 2] = 0
        elif id_str[i] == '1':
            for j in range(k):
                for s in range(NUM_ROWS):
                    stamped_frame[-s-1, i * k + j, 0] = 255
                    stamped_frame[-s-1, i * k + j, 1] = 255
                    stamped_frame[-s-1, i * k + j, 2] = 255

    stamped_frame = np.uint8(stamped_frame)
    final_frame = av.VideoFrame.from_ndarray(stamped_frame)
    final_frame.pts = frame_pts
    final_frame.time_base = frame_time_base
    return final_frame


def destamp_frame(frame):
    """ retrieve frame index and original frame from barcoded frame
    """
    frame_array = frame.to_rgb().to_ndarray()
    k = frame_array.shape[1] // NUMBER_OF_BITS
    destamped_frame = frame_array[:-NUM_ROWS]

    frame_id = frame_array[-NUM_ROWS:, :, :]
    frame_id = frame_id.mean(0)
    frame_id = frame_id[frame_array.shape[1] - k*NUMBER_OF_BITS:, :]

    frame_id = np.reshape(frame_id, [NUMBER_OF_BITS, k, 3])
    frame_id = frame_id.mean(axis=(1,2))

    frame_id = (frame_id > (frame_id.max() + frame_id.min()) / 2 * 1.2 ).astype(int)
    frame_id = ((2 ** (NUMBER_OF_BITS - 1 - np.arange(NUMBER_OF_BITS))) * frame_id).sum()
    frame_id = frame_id - 1

    destamped_frame = np.uint8(destamped_frame)
    final_frame = av.VideoFrame.from_ndarray(destamped_frame)
    return final_frame, frame_id


def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset, timing_enabled, 
                   save_visualizations_as_images, experiment_name, reference_frame_update_freq=None, profile=False, netadapt_checkpoint=None):
    """
    Netadapt checkpoint vs regular checkpoint
    Then netadapt checkpoint is used to load just the netadapted `generator` and `kp_detector`
    although it is frozen so both the netadapt and regular version contains the same
    `kp_detector`.
    Regular checkpoint is used to load the rest of the model.
    """
    """ reconstruct driving frames for each video in the dataset using the first frame
        as a source frame. Config specifies configuration details, while timing 
        determines whether to time the functions on a gpu or not """
    global ssim_correlation_file
    global reference_frame_list
    png_dir = os.path.join(log_dir, 'png')
    visualization_dir = os.path.join(log_dir, 'visualization')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator_params = config['model_params']['generator_params']
    generator_type = generator_params.get('generator_type', 'occlusion_aware')
    lr_size = generator_params.get('lr_size', 64)
    print('reference_frame_update_freq', reference_frame_update_freq, 'lr_size', lr_size)

    train_params = config['train_params']
    target_bitrate = train_params.get('target_bitrate', 1000000)
    quantizer_level = train_params.get('quantizer_level', -1)
    encoder_in_training = train_params.get('encode_video_for_training', False)
    codec_params = train_params.get('codec', 'vp8')
    print(f'Encoding using {codec_params}')
    
    choose_reference_frame = False
    use_same_tgt_ref_quality = False
    
    if generator_type not in ['vpx', 'bicubic', 'swinir']:
        if checkpoint is not None:
            dense_motion = generator.dense_motion_network if generator_type == 'occlusion_aware' else None
            Logger.load_cpk(checkpoint, generator=generator, 
                    kp_detector=kp_detector, device=device, 
                    dense_motion_network=dense_motion, generator_type=generator_type, reconstruction=True)

    # Manually force the generator and keypoint netadapted model weights
    # into the network. It does reload the keypoint detector, but since
    # both checkpoints contain tthe same kp detector, only the generator
    # is changed.
    if netadapt_checkpoint is not None:
        state_dict = torch.load(netadapt_checkpoint)
        set_gen_module(generator, state_dict)
        set_keypoint_module(kp_detector, state_dict)
        print('reloaded params')

    if checkpoint is None and netadapt_checkpoint is None:
        raise AttributeError('Checkpoint should be specified for reconstruction')

    # get number of model parameters and mac stats
    if profile:
        if generator_type == 'swinir':
            get_model_info(log_dir, kp_detector, generator.model)
        else:
            get_model_info(log_dir, kp_detector, generator)

        image_size = config['dataset_params']['frame_shape'][0]
        get_model_macs(log_dir, generator, kp_detector, device, lr_size, image_size)
        return
    
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
    vgg_face_model = VggFace16()
    original_lpips = lpips.LPIPS(net='vgg')
    if torch.cuda.is_available():
        if generator is not None and generator_type != 'swinir':
            generator = DataParallelWithCallback(generator)
        if kp_detector is not None:
            kp_detector = DataParallelWithCallback(kp_detector)
        vgg_model = vgg_model.cuda()
        vgg_face_model = vgg_face_model.cuda()
        original_lpips = original_lpips.cuda()
 
    loss_fn_vgg = vgg_model.compute_loss
    face_lpips = vgg_face_model.compute_loss

    if generator is not None and generator_type != 'swinir':
        generator.eval()
    if kp_detector is not None:
        kp_detector.eval()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for it, x in tqdm(enumerate(dataloader)):
        updated_src = 0
        reference_stream = [0]
        lr_stream = [0]
        
        if codec_params == 'vp9':
            hr_encoder, lr_encoder = Vp9Encoder(), Vp9Encoder()
            hr_decoder, lr_decoder = Vp9Decoder(), Vp9Decoder()
        else:
            hr_encoder, lr_encoder = Vp8Encoder(), Vp8Encoder()
            hr_decoder, lr_decoder = Vp8Decoder(), Vp8Decoder()

        with torch.no_grad():
            predictions = []
            visualizations = []
            visual_metrics = []
            video_name = x['video_path'][0]
            print('doing video', video_name)
            visualizations = []
            video_duration = get_video_duration(video_name)

            driving_times, generator_times = [], []
            source_time = 0

            container = av.open(file=video_name, format=None, mode='r')
            stream = container.streams.video[0]

            frame_idx = 0
            for av_frame in container.decode(stream):
                # ground-truth
                frame = av_frame.to_rgb().to_ndarray()
                driving = frame_to_tensor(img_as_float32(frame), device)
                
                # get LR video frame
                driving_lr = resize_tensor_to_array(driving, lr_size, device)

                if frame_idx in special_frames_list and SAVE_LR_FRAMES and generator_type != 'vpx':
                    np.save(os.path.join(visualization_dir,
                        'sender_lr_frame_%05d.npy' % av_frame.index), driving_lr)

                driving_lr_av = av.VideoFrame.from_ndarray(driving_lr)
                driving_lr_av.pts = av_frame.pts
                driving_lr_av.time_base = av_frame.time_base
                
                if encoder_in_training:
                    driving_lr_av, driving_lr, compressed_tgt = get_frame_from_video_codec(driving_lr_av,
                            av_frame.index, lr_encoder, lr_decoder, quantizer_level, target_bitrate)
                else:
                    compressed_tgt = 0

                driving_lr_array = driving_lr
                if frame_idx in special_frames_list and SAVE_LR_FRAMES and generator_type != 'vpx':
                    np.save(os.path.join(visualization_dir,
                        'receiver_lr_frame_%05d.npy' % av_frame.index), driving_lr_array)

                driving_lr = frame_to_tensor(img_as_float32(driving_lr), device)
                
                # for use as source frame
                if generator_type == 'vpx':
                    decoded_frame_av, decoded_frame, compressed_src = get_frame_from_video_codec(av_frame,
                            av_frame.index, hr_encoder, hr_decoder, quantizer_level, target_bitrate)
                elif encoder_in_training:
                    decoded_frame_av, decoded_frame, compressed_src = get_frame_from_video_codec(av_frame,
                            av_frame.index, hr_encoder, hr_decoder, 32)
                else:
                    decoded_frame_av = av_frame
                    decoded_frame = frame
                    compressed_src = 0

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
                        source_time = start.elapsed_time(end)

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
 
                if kp_detector is not None:
                    start.record()
                    if generator_params.get('use_lr_video', False):
                        kp_driving = kp_detector(driving_lr)
                    else:
                        kp_driving = kp_detector(driving)
                    end.record()
                    torch.cuda.synchronize()
                    driving_times.append(start.elapsed_time(end))
                else:
                    driving_times.append(0)
                
                start.record()
                if generator_type in ['occlusion_aware', 'split_hf_lf', 'student_occlusion_aware']:
                    out = generator(source, kp_source=kp_source, \
                            kp_driving=kp_driving, update_source=update_source, driving_lr=driving_lr)

                    if use_same_tgt_ref_quality:
                        ref_out = generator(driving, kp_source=kp_driving, \
                            kp_driving=kp_driving, update_source=True, driving_lr=driving_lr)
                    
                    if generator_params.get('use_lr_video', False):
                        lr_stream.append(compressed_tgt)
                    else:
                        lr_stream.append(KEYPOINT_FIXED_PAYLOAD_SIZE)

                elif generator_type == 'bicubic':
                    upsampled_frame = driving_lr_av.reformat(width=driving.shape[2], \
                            height=driving.shape[3],\
                            interpolation='BICUBIC').to_rgb().to_ndarray()
                    out = {'prediction': frame_to_tensor(img_as_float32(upsampled_frame), device)}
                    lr_stream.append(compressed_tgt)
                    reference_stream.append(0)
                
                elif generator_type == 'vpx':
                    out = {'prediction': decoded_tensor}
                    lr_stream.append(0)
                    reference_stream.append(compressed_src)

                elif generator_type == 'swinir':
                    predicted_array = generator.predict_with_lr_video(driving_lr_array)
                    out = {'prediction': frame_to_tensor(img_as_float32(predicted_array), device)}
                    lr_stream.append(compressed_tgt)
                    reference_stream.append(0)
                else:
                    out = generator(driving_lr)
                    lr_stream.append(compressed_tgt)
                
                end.record()
                torch.cuda.synchronize()
                generator_times.append(start.elapsed_time(end))

                out['prediction'] = torch.clamp(out['prediction'], min=0, max=1)
                
                """
                ssim = piq.ssim(driving, out['prediction'], data_range=1.).data.cpu().numpy().flatten()[0]
                if use_same_tgt_ref_quality and (generator_type in ['occlusion_aware', 'split_hf_lf']):
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
                """
                                
                if kp_detector is not None:
                    out['kp_source'] = kp_source
                    out['kp_driving'] = kp_driving
                    if 'sparse_deformed' in out:
                        del out['sparse_deformed']

                if frame_idx % 200 == 0:
                    print(f'finished {frame_idx} frames, updated src: {updated_src}')
                
                if frame_idx in special_frames_list or generate_video_visualizations:
                    if generator_type not in ['vpx', 'bicubic']:
                        v = Visualizer(**config['visualizer_params']).visualize(source=source,
                                                                                driving=driving, out=out)
                    else:
                        v = out['prediction'].data.cpu().numpy().transpose(0, 2, 3, 1)[0]

                    if generate_video_visualizations:
                        v = (255 * v).astype(np.uint8)
                        visualizations.append(v)
                        if frame_idx % 1000 == 0:
                            image_name = f"{x['name'][0]}_{frame_idx}_{config['reconstruction_params']['format']}"
                            imageio.mimsave(os.path.join(log_dir, image_name), visualizations, fps=30)
                            print(f'saving {frame_idx} frames')
                            visualizations = []
                    else:
                        frame_name = f"{x['name'][0]}_frame{frame_idx}.npy"
                        frame_file = open(os.path.join(visualization_dir, frame_name), 'wb')
                        np.save(frame_file, v)
                        frame_file.close()
                
                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())
                visual_metrics.append(Logger.get_visual_metrics(out['prediction'], driving, \
                        loss_fn_vgg, original_lpips, face_lpips))
                
            container.close()
            print('total frames', frame_idx, 'updated src', updated_src)
            ref_br = get_bitrate(reference_stream, video_duration)
            lr_br = get_bitrate(lr_stream, video_duration)
            
            psnr, ssim, lpips_val, ssim_db, orig_lpips, face_lpips_val = \
                    get_avg_visual_metrics(visual_metrics)
            metrics_file.write(f"{x['name'][0]} PSNR: {psnr}, SSIM: {ssim}, SSIM_DB: {ssim_db}, " + \
                    f'LPIPS: {lpips_val}, ' +
                    f'Standard LPIPS: {orig_lpips}, Face LPIPS: {face_lpips_val}, ' + 
                    f'Reference: {ref_br:.3f}Kbps, LR: {lr_br:.3f}Kbps, ' + 
                    f'KP extraction: {np.average(driving_times)}ms, ' +
                    f'Generator: {np.average(generator_times)}ms \n')
            metrics_file.flush()

            if it == 0:
                frame_metrics_file.write('video_num,frame,psnr,ssim,ssim_db,lpips,orig_lpips,face_lpips,' + \
                        'kp_time,gen_time,reference_kbps,lr_kbps\n')
            for i, (m, d, g) in enumerate(zip(visual_metrics, driving_times, generator_times)):
                frame_metrics_file.write(f"{it + 1},{i},{m['psnr'][0]},{m['ssim']},{m['ssim_db']}," + \
                            f"{m['lpips']},{m['orig_lpips']},{m['face_lpips']},{d},{g},{ref_br},{lr_br}\n")
            frame_metrics_file.flush()
            
            print('source keypoints:', source_time, 'driving:', np.average(driving_times), \
                'generator:', np.average(generator_times))

    print('Reconstruction loss: %s' % np.mean(loss_list))
    metrics_file.write('Reconstruction loss: %s\n' % np.mean(loss_list))
    metrics_file.close()
    frame_metrics_file.close()
    ssim_correlation_file.close()

