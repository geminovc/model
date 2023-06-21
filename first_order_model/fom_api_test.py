from first_order_model.fom_wrapper import FirstOrderModel
import imageio 
import numpy as np
import time
from argparse import ArgumentParser
from frames_dataset import get_num_frames, get_frame
import torch
import piq
from skimage import img_as_float32
from first_order_model.modules.model import Vgg19, VggFace16
from reconstruction import *
from utils import get_main_config_params
import lpips

parser = ArgumentParser()
parser.add_argument("--config", 
                    default="config/paper_configs/resolution512_with_hr_skip_connections.yaml",
                    help="path to config")
parser.add_argument("--checkpoint",
                    default='None',
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
                    type=int, default=None,
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

main_configs = get_main_config_params(args.config)
generator_type = main_configs['generator_type']
use_lr_video = main_configs['use_lr_video']
lr_size = main_configs['lr_size']
print(main_configs)

video_duration = get_video_duration(args.video_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# model initialization and warm-up
if generator_type not in ['vpx', 'bicubic']:
    model = FirstOrderModel(args.config, args.checkpoint)
    source = np.random.rand(model.get_shape()[0], model.get_shape()[1], model.get_shape()[2])
    source_kp, _= model.extract_keypoints(source)
    if use_lr_video:
        source_tensor = frame_to_tensor(img_as_float32(source), device)
        source_lr = resize_tensor_to_array(source_tensor, lr_size, device)

    model.update_source(0, source, source_kp)
    for _ in range(100):
        if use_lr_video:
            _ = model.predict_with_lr_video(source_lr)
        else:
            source_kp['source_index'] = 0
            _ = model.predict(source_kp)
    model.reset()
    get_model_info(args.log_dir, model.kp_detector, model.generator)

choose_reference_frame = False #TODO
use_same_tgt_ref_quality = False #TODO
timing_enabled = False if generator_type in ['vpx', 'bicubic'] else True
reference_frame_list = []
predictions = []
visual_metrics = []
loss_list = []
updated_src = 0
reference_stream = []
lr_stream = []
metrics_file = open(os.path.join(args.log_dir, args.output_name + '_metrics_summary.txt'), 'wt')
frame_metrics_file = open(os.path.join(args.log_dir, args.output_name + '_per_frame_metrics.txt'), 'wt')
write_in_file(frame_metrics_file, 'frame,psnr,ssim,ssim_db,lpips,orig_lpips,face_lpips\n')

vgg_model = Vgg19()
vgg_face_model = VggFace16()
original_lpips = lpips.LPIPS(net='vgg')
if torch.cuda.is_available():
    vgg_model = vgg_model.cuda()
    vgg_face_model = vgg_face_model.cuda()
    original_lpips = original_lpips.cuda()

loss_fn_vgg = vgg_model.compute_loss
face_lpips = vgg_face_model.compute_loss

start = torch.cuda.Event(enable_timing=timing_enabled)
end = torch.cuda.Event(enable_timing=timing_enabled)
hr_encoder, lr_encoder = Vp8Encoder(), Vp8Encoder()
hr_decoder, lr_decoder = Vp8Decoder(), Vp8Decoder()
container = av.open(file=args.video_path, format=None, mode='r')
stream = container.streams.video[0]

if timing_enabled:
    generator_times, update_source_times = [], []

with torch.no_grad():
    frame_idx = 0
    for av_frame in container.decode(stream):
        # ground-truth
        frame = av_frame.to_rgb().to_ndarray()
        driving = frame
        driving_tensor = frame_to_tensor(img_as_float32(frame), device)
        driving_lr = resize_tensor_to_array(driving_tensor, lr_size, device)

        driving_lr_av = av.VideoFrame.from_ndarray(driving_lr)
        driving_lr_av.pts = av_frame.pts
        driving_lr_av.time_base = av_frame.time_base

        if args.encode_lr:
            _, driving_lr, compressed_tgt = get_frame_from_video_codec(driving_lr_av, frame_idx, lr_encoder,
                                                                lr_decoder, args.lr_quantizer)
        # for use as source frame
        if args.encode_hr:
            _, decoded_array, compressed_src = get_frame_from_video_codec(av_frame, frame_idx, hr_encoder,
                                                                hr_decoder, args.hr_quantizer)
        else:
            decoded_array = driving

        if generator_type in ['occlusion_aware', 'split_hf_lf']:
            update_source = False if not use_same_tgt_ref_quality else True
            # keypoint detector
            if model.kp_detector is not None:
                if frame_idx == 0:
                    source = decoded_array
                    kp_source, _ = model.extract_keypoints(source)
                    update_source = True

                if args.reference_frame_update_freq is not None:
                    if frame_idx % args.reference_frame_update_freq == 0:
                        source = decoded_array
                        kp_source, _ = model.extract_keypoints(source)
                        update_source = True

                elif choose_reference_frame: #TODO
                    # runs at sender, so use frame prior to encode/decode pipeline
                    cur_frame = frame_to_tensor(frame, device)
                    cur_kp = model.kp_detector(cur_frame)

                    frame_reuse, source, kp_source = find_best_reference_frame(cur_frame, cur_kp, \
                        video_num=it+1, frame_idx=frame_idx)

                if update_source:
                    start.record()
                    model.update_source(len(model.source_frames), source, kp_source)
                    end.record()
                    torch.cuda.synchronize()
                    reference_frame_list.append((source, kp_source))
                    if timing_enabled:
                        update_source_times.append(start.elapsed_time(end))
                    if args.encode_hr:
                        reference_stream.append(compressed_src)

            else:
                # default if there's no KP based method
                source = decoded_array

            if use_lr_video:
                if args.encode_lr:
                    lr_stream.append(compressed_tgt)
            else:
                lr_stream.append(KEYPOINT_FIXED_PAYLOAD_SIZE)

            start.record()
            if use_lr_video:
                prediction = model.predict_with_lr_video(driving_lr)
            else:
                target_kp, source_index = model.extract_keypoints(driving)
                target_kp['source_index'] = source_index
                prediction = model.predict(target_kp)

            if use_same_tgt_ref_quality:
                ref_out = model.generator(driving, kp_source=kp_driving, \
                    kp_driving=kp_driving, update_source=True, driving_lr=driving_lr)

        elif generator_type == 'bicubic':
            driving_lr_tensor = frame_to_tensor(img_as_float32(driving_lr), device)
            prediction = resize_tensor_to_array(driving_lr_tensor, driving.shape[1], device, mode='bicubic')
            if args.encode_lr:
                lr_stream.append(compressed_tgt)

        elif generator_type == 'vpx':
            prediction = decoded_array
            if args.encode_hr:
                reference_stream.append(compressed_src)

        else:
            # generator_type could be "only_upsampler"
            prediction = model.predict_with_lr_video(driving_lr)
            if args.encode_lr:
                lr_stream.append(compressed_tgt)

        end.record()
        torch.cuda.synchronize()
        if timing_enabled:
            generator_times.append(start.elapsed_time(end))

        prediction_tensor = frame_to_tensor(img_as_float32(prediction), device)
        loss_list.append(torch.abs(prediction_tensor - driving_tensor).mean().cpu().numpy())
        visual_metrics.append(Logger.get_visual_metrics(prediction_tensor, driving_tensor, loss_fn_vgg, original_lpips, face_lpips))
        predictions.append(prediction)
        if frame_idx % 100 == 0:
            print('total frames', frame_idx, 'updated src', updated_src)
        frame_idx += 1

imageio.mimsave(os.path.join(args.log_dir, 
                args.output_name + f'_freq{args.reference_frame_update_freq}.mp4'),
                predictions, fps = int(args.output_fps))

ref_br = get_bitrate(reference_stream, video_duration)
lr_br = get_bitrate(lr_stream, video_duration)

for i, m in enumerate(visual_metrics):
    write_in_file(frame_metrics_file, f"{i},{m['psnr'][0]},{m['ssim']}," +
                f"{m['ssim_db']},{m['lpips']},{m['orig_lpips']},{m['face_lpips']}\n")
frame_metrics_file.close()

psnr, ssim, lpips_val, ssim_db, orig_lpips_val, face_lpips_val = get_avg_visual_metrics(visual_metrics)
metrics_report = f'reference_frame_update_freq {args.reference_frame_update_freq}'
if args.encode_hr:
    metrics_report += f', hr_quantizer: {args.hr_quantizer}'

if args.encode_lr:
    metrics_report += f', lr_quantizer: {args.lr_quantizer}'

metrics_report += f'\nPSNR: {psnr}, SSIM: {ssim}, SSIM_DB: {ssim_db}, LPIPS: {lpips_val}, ' +\
    f'ORIG_LPIPS: {orig_lpips_val}, FACE_LPIPS: {face_lpips_val} ' +\
    f'Reference: {ref_br:.3f}Kbps, LR: {lr_br:.3f}Kbps \n' +\
    f'Reconstruction loss: {np.mean(loss_list)}\n'

if timing_enabled:
    metrics_report += f'update source: {np.average(update_source_times)}, ' +\
    f'generator: {np.average(generator_times)}'

write_in_file(metrics_file, metrics_report)
metrics_file.close()

