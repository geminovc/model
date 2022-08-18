from first_order_model.fom_wrapper import FirstOrderModel
import imageio 
import numpy as np
import time
from argparse import ArgumentParser
from frames_dataset import get_num_frames, get_frame
import torch
import piq
from skimage import img_as_float32
from first_order_model.modules.model import Vgg19
from reconstruction import *

parser = ArgumentParser()
parser.add_argument("--config", 
                    default="config/paper_configs/resolution512_with_hr_skip_connections.yaml",
                    help="path to config")
parser.add_argument("--checkpoint",
                    required=True,
                    help="path to the checkpoints")
parser.add_argument("--video-path",
                    default="512_kayleigh_10_second_0_1.mp4",
                    help="path to the video")
parser.add_argument("--log-dir",
                    default="./fom_api_test",
                    help="directory to save the results")
parser.add_argument("--output-name",
                    default="prediction",
                    help="name of the output file to be saved")
parser.add_argument("--output-fps",
                    default=30,
                    help="fps of the final video")
parser.add_argument("--reference_frame_update_freq",
                    default=None,
                    help="reference update frequency")
parser.add_argument("--hr-quantizer",
                    type=int, default=16,
                    help="quantizer to compress high-res video stream with")
parser.add_argument("--encode-hr",
                    action='store_true',
                    help="encode high-res video stream with vpx")
parser.add_argument("--lr-quantizer",
                    type=int, default=32,
                    help="quantizer to compress low-res video stream with")
parser.add_argument("--encode-lr",
                    action='store_true',
                    help="encode low-res video stream with vpx")
parser.set_defaults(verbose=False)
args = parser.parse_args()

def get_lr_array(input_tensor, lr_size, device):
    lr_array = F.interpolate(input_tensor, lr_size).data.cpu().numpy()
    lr_array = np.transpose(lr_array, [0, 2, 3, 1])[0]
    lr_array *= 255
    lr_array = lr_array.astype(np.uint8)
    return lr_array

def write_in_file(input_file, info):
    input_file.write(info)
    input_file.flush()

model = FirstOrderModel(args.config, args.checkpoint)
video_name = args.video_path
video_duration = get_video_duration(video_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
source = np.random.rand(model.get_shape()[0], model.get_shape()[1], model.get_shape()[2])
source_tensor = frame_to_tensor(img_as_float32(source), device)
source_kp, _= model.extract_keypoints(source)
model.update_source(0, source, source_kp)
use_lr_video = True
generator_type = 'occlusion_aware'
lr_size = 256
if use_lr_video:
    source_lr = get_lr_array(source_tensor, lr_size, device)

choose_reference_frame = False
use_same_tgt_ref_quality = False
timing_enabled = True
reference_frame_list = []
predictions = []
visual_metrics = []
loss_list = []
updated_src = 0
reference_stream = []
lr_stream = []

# warm-up
for _ in range(1):
    if use_lr_video:
        _ = model.predict_with_lr_video(source_lr)
    else:
        source_kp['source_index'] = 0
        _ = model.predict(source_kp)
model.reset()

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

metrics_file = open(os.path.join(args.log_dir, args.output_name + '_metrics_summary.txt'), 'wt')
frame_metrics_file = open(os.path.join(args.log_dir, args.output_name + '_per_frame_metrics.txt'), 'wt')
write_in_file(frame_metrics_file, 'frame,psnr,ssim,ssim_db,lpips\n')
get_model_info(args.log_dir, model.kp_detector, model.generator)
vgg_model = Vgg19()

if torch.cuda.is_available():
    vgg_model = vgg_model.cuda()
loss_fn_vgg = vgg_model.compute_loss

start = torch.cuda.Event(enable_timing=timing_enabled)
end = torch.cuda.Event(enable_timing=timing_enabled)
hr_encoder, lr_encoder = Vp8Encoder(), Vp8Encoder()
hr_decoder, lr_decoder = Vp8Decoder(), Vp8Decoder()
container = av.open(file=video_name, format=None, mode='r')
stream = container.streams.video[0]

if timing_enabled:
    driving_times, generator_times, visualization_times = [], [], []

with torch.no_grad():
    frame_idx = 0
    for av_frame in container.decode(stream):
        # ground-truth
        frame = av_frame.to_rgb().to_ndarray()
        driving = frame #frame_to_tensor(img_as_float32(frame), device)
        driving_lr = get_lr_array(frame_to_tensor(img_as_float32(frame), device), lr_size, device)

        driving_lr_av = av.VideoFrame.from_ndarray(driving_lr)
        driving_lr_av.pts = av_frame.pts
        driving_lr_av.time_base = av_frame.time_base

        if args.encode_lr:
            driving_lr, compressed_tgt = get_frame_from_video_codec(driving_lr_av, lr_encoder,
                                                                lr_decoder, args.lr_quantizer)
           # driving_lr = frame_to_tensor(img_as_float32(driving_lr), device)
        #else:
        #    driving_lr = frame_to_tensor(driving_lr, device)

        # for use as source frame
        if args.encode_hr:
            decoded_tensor, compressed_src = get_frame_from_video_codec(av_frame, hr_encoder,
                                                                hr_decoder, args.hr_quantizer)
           # decoded_tensor = frame_to_tensor(img_as_float32(decoded_frame), device)
        else:
            decoded_tensor = driving

        update_source = False if not use_same_tgt_ref_quality else True
        if model.kp_detector is not None:
            if frame_idx == 0:
                source = decoded_tensor
                start.record()
                kp_source, _ = model.extract_keypoints(source)
                model.update_source(len(model.source_frames), source, kp_source)
                end.record()
                torch.cuda.synchronize()
                update_source = True
                reference_frame_list.append((source, kp_source))
                if timing_enabled:
                    source_time = start.elapsed_time(end)
                if args.encode_hr:
                    reference_stream.append(compressed_src)

            if args.reference_frame_update_freq is not None:
                if frame_idx % args.reference_frame_update_freq == 0:
                    source = decoded_tensor
                    kp_source = model.kp_detector(source)
                    update_source = True
                    if args.encode_hr:
                        reference_stream.append(compressed_src)

            elif choose_reference_frame: #TODO
                # runs at sender, so use frame prior to encode/decode pipeline
                cur_frame = frame_to_tensor(frame, device)
                cur_kp = model.kp_detector(cur_frame)

                frame_reuse, source, kp_source = find_best_reference_frame(cur_frame, cur_kp, \
                    video_num=it+1, frame_idx=frame_idx)
        else:
            # default if there's no KP based method
            source = decoded_tensor

        frame_idx += 1
        if args.encode_lr:
            if use_lr_video:
                lr_stream.append(compressed_tgt)
            else:
                lr_stream.append(KEYPOINT_FIXED_PAYLOAD_SIZE)

        #if model.kp_detector is not None:
        #    start.record()
        #    if use_lr_video:
        #        kp_driving = model.kp_detector(driving_lr)
        #    else:
        #        kp_driving = model.kp_detector(driving)
        #    end.record()
        #    torch.cuda.synchronize()
        #    if timing_enabled:
        #        driving_times.append(start.elapsed_time(end))

        start.record()
        if generator_type in ['occlusion_aware', 'split_hf_lf']:
            print("here")
            prediction = model.predict_with_lr_video(driving_lr)
            #out = model.generator(source, kp_source=kp_source, \
            #        kp_driving=kp_driving, update_source=update_source, driving_lr=driving_lr)

            if use_same_tgt_ref_quality:
                ref_out = model.generator(driving, kp_source=kp_driving, \
                    kp_driving=kp_driving, update_source=True, driving_lr=driving_lr)

        elif generator_type == "bicubic":
            out = {'prediction': F.interpolate(driving_lr, source.shape[2], mode='bicubic')}
            if args.encode_lr:
                lr_stream.append(compressed_tgt)

        elif generator_type == "vpx":
            out = {'prediction': decoded_tensor}
            if args.encode_hr:
                reference_stream.append(compressed_src)

        else:
            out = model.generator(driving_lr)
            if args.encode_lr:
                lr_stream.append(compressed_tgt)

        end.record()
        torch.cuda.synchronize()
        if timing_enabled:
            generator_times.append(start.elapsed_time(end))

        #out['prediction'] = torch.clamp(out['prediction'], min=0, max=1)
        
        '''
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
        '''
        loss_list.append(torch.abs(frame_to_tensor(img_as_float32(prediction), device) - frame_to_tensor(img_as_float32(driving), device)).mean().cpu().numpy())
        visual_metrics.append(Logger.get_visual_metrics(frame_to_tensor(img_as_float32(prediction), device), frame_to_tensor(img_as_float32(driving), device), loss_fn_vgg))
        predictions.append(prediction)
        if frame_idx % 100 == 0:
            print('total frames', frame_idx, 'updated src', updated_src)
        if frame_idx > 60:
            break

imageio.mimsave(os.path.join(args.log_dir, 
                args.output_name + f'_freq{args.reference_frame_update_freq}.mp4'),
                predictions, fps = int(args.output_fps))

ref_br = get_bitrate(reference_stream, video_duration)
lr_br = get_bitrate(lr_stream, video_duration)

for i, m in enumerate(visual_metrics):
    write_in_file(frame_metrics_file, f'{i},{m["psnr"][0]},{m["ssim"]},' + 
                f'{m["ssim_db"]},{m["lpips"]}\n')
frame_metrics_file.close()

psnr, ssim, lpips_val, ssim_db = get_avg_visual_metrics(visual_metrics)
write_in_file(metrics_file, f'PSNR: {psnr}, SSIM: {ssim}, SSIM_DB: {ssim_db}, ' + 
    f'LPIPS: {lpips_val}, Reference: {ref_br:.3f}Kbps, LR: {lr_br:.3f}Kbps \n')
write_in_file(metrics_file, f'Reconstruction loss: {np.mean(loss_list)}\n')
metrics_file.close()

