""" This script measures the run time of onnx vs pytorch vs deepsparse conversions of FOM
    per frame with running them on a single video. The script aslo stores visual metrics along
    with timings in a csv file.
"""
from first_order_model.fom_wrapper import FirstOrderModel
import original_bilayer.examples.utils as metrics_utils
import numpy as np
import onnxruntime
import torch
import time
import imageio
import argparse
import csv
import platform
if platform.system() == 'Linux':
    from deepsparse import compile_model


parser = argparse.ArgumentParser(description='Get timing and visual metrics of different FOM conversions')
parser.add_argument('--video_path',
                        type = str,
                        default = '../short_test_video.mp4',
                        help = 'path to the video')
parser.add_argument('--config_path',
                        type = str,
                        default = '../config/api_sample.yaml',
                        help = 'path to the config file')
parser.add_argument('--onnx_path',
                        type = str,
                        default = './',
                        help = 'path to the onnx files')
parser.add_argument('--csv_file_name',
                       type = str,
                       required = True,
                       help = 'name of csv file to write results out to')


def run_kp_detector(net_type, driving, model=None, onnx_session=None):
    if net_type == 'pytorch':
        start = time.time()
        target_kp = model.extract_keypoints(driving)
        kp_time = time.time() - start
        target_jacobian = None
    else:
        onnx_session_kp_input = {'source': np.array(np.transpose(driving[None, :], (0, 3, 1, 2)), dtype=np.float32)/255}
        start = time.time()
        target_kp, target_jacobian = onnx_session.run(None, onnx_session_kp_input)
        kp_time = time.time() - start

    return kp_time, target_kp, target_jacobian


def run_generator(net_type, torch_inputs=None, ort_inputs=None, model=None, onnx_session=None):
    if net_type == 'pytorch':
        start = time.time()
        frame_next = model.predict(torch_inputs)
        gen_time = time.time() - start
    else:
        start = time.time()
        ort_outs = onnx_session.run(None, ort_inputs)
        gen_time = time.time() - start
        frame_next = np.transpose(ort_outs[0].squeeze(), (1, 2, 0))

    return gen_time, frame_next


def run_deepsparse(inputs, engine):
    start = time.time()
    outputs = engine.run(inputs)
    gen_time = time.time() - start
    frame_next = np.transpose(outputs[0].squeeze(), (1, 2, 0))
    return gen_time, frame_next


def run_inference(video_name, model_type='onnx', csv_file_name='timings.csv',
                  config_path='../config/api_sample.yaml', onnx_path="./"):
    video_array = np.array(imageio.mimread(video_name))
    source = video_array[0, :, :, :]

    if model_type == 'pytorch':
        # set up model
        model = FirstOrderModel(config_path)
        source_kp = model.extract_keypoints(source)
        model.update_source(source, source_kp)
    elif model_type == 'pytorch + kp_onnx':
        model = FirstOrderModel(config_path)
        onnx_session_kp_extractor = onnxruntime.InferenceSession(onnx_path + "fom_kp.onnx")
        source_reshaped = np.array(np.transpose(source[None, :], (0, 3, 1, 2)), dtype=np.float32)/255
        onnx_session_kp_input = {'source': source_reshaped}
        source_kp, source_jacobian = onnx_session_kp_extractor.run(None, onnx_session_kp_input)
        model.update_source(source, {'keypoints': np.squeeze(source_kp), 'jacobians': source_jacobian})
    elif model_type == 'pytorch + generator_onnx':
        model = FirstOrderModel(config_path)
        onnx_session_generator = onnxruntime.InferenceSession(onnx_path + "fom_gen.onnx")
        source_kp = model.extract_keypoints(source)
        model.update_source(source, source_kp)
    elif model_type == 'onnx' or model_type == 'deepsparse':
        onnx_session_generator = onnxruntime.InferenceSession(onnx_path + "fom_gen.onnx")
        onnx_session_kp_extractor = onnxruntime.InferenceSession(onnx_path + "fom_kp.onnx")
        source_reshaped = np.array(np.transpose(source[None, :], (0, 3, 1, 2)), dtype=np.float32)/255
        onnx_session_kp_input = {'source': source_reshaped}
        source_kp, source_jacobian = onnx_session_kp_extractor.run(None, onnx_session_kp_input)
        if model_type == 'deepsparse':
            batch_size = 1
            # kp_engine = compile_model(onnx_path + "fom_kp.onnx", batch_size)
            gen_engine = compile_model(onnx_path + "fom_gen.onnx", batch_size)

    predictions, psnr_values, ssim_values, lpips_values, kp_times, gen_times = [], [], [], [], [], []

    for i in range(1, len(video_array) - 1):
        driving = video_array[i, :, :, :]

        if model_type == 'pytorch':
            kp_time, target_kp, _ = run_kp_detector('pytorch', driving, model)
            gen_time, frame_next = run_generator('pytorch', torch_inputs=target_kp, model=model)
        elif model_type == 'pytorch + kp_onnx':
            kp_time, target_kp, target_jacobian = run_kp_detector('onnx', driving, onnx_session=onnx_session_kp_extractor)
            torch_inputs = {'keypoints': np.squeeze(target_kp)}
            gen_time, frame_next = run_generator('pytorch', torch_inputs=torch_inputs, model=model)
        elif model_type == 'pytorch + generator_onnx':
            kp_time, target_kp, _ = run_kp_detector('pytorch', driving, model)
            target_reshaped = np.array(np.transpose(source[None, :], (0, 3, 1, 2)), dtype=np.float32)/255
            ort_inputs = {'source_image': target_reshaped,
                          'kp_driving_v': target_kp['keypoints'][None, :],
                          'kp_source_v': source_kp['keypoints'][None, :]}
            gen_time, frame_next = run_generator('onnx', ort_inputs=ort_inputs, onnx_session=onnx_session_generator)
        elif model_type == 'onnx' or model_type == 'deepsparse':
            kp_time, target_kp, _ = run_kp_detector('onnx', driving, onnx_session=onnx_session_kp_extractor)
            if model_type == 'onnx':
                target_reshaped = np.array(np.transpose(source[None, :], (0, 3, 1, 2)), dtype=np.float32)/255
                ort_inputs = {'source_image': target_reshaped,
                              'kp_driving_v': target_kp,
                              'kp_source_v': source_kp}
                gen_time, frame_next = run_generator('onnx', ort_inputs=ort_inputs, onnx_session=onnx_session_generator)
            else:
                target_reshaped = np.ascontiguousarray(np.array(np.transpose(source[None, :],
                                                       (0, 3, 1, 2)), dtype=np.float32)/255)
                inputs = [target_reshaped, target_kp, source_kp]
                gen_time, frame_next = run_deepsparse(inputs, gen_engine)

        gen_times.append(gen_time)
        kp_times.append(kp_time)
        predictions.append(frame_next)
        psnr_value, ssim_value, lpips_value = metrics_utils.compute_metric_for_files((255 * frame_next).astype(np.uint8), driving)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        lpips_values.append(lpips_value)
    
    imageio.mimsave(f'prediction_{model_type}.mp4', predictions)
    num_frames = len(kp_times)
    print(f'Average prediction time for {model_type}:', sum(gen_times)/len(gen_times))

    with open(model_type + '_' + csv_file_name, 'w', encoding='UTF8') as f:
        header = ['min_kpD_time', 'mean_kpD_time', 'max_kpD_time','min_gen_time', 'mean_gen_time',
                  'max_gen_time', 'min_psnr', 'mean_psnr', 'max_psnr', 'min_ssim', 'mean_ssim', 
                  'max_ssim', 'min_lpips', 'mean_lpips', 'max_lpips']
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow([min(kp_times), sum(kp_times)/num_frames, max(kp_times), min(gen_times),
                         sum(gen_times)/num_frames, max(gen_times), min(psnr_values),
                         sum(psnr_values)/num_frames, max(psnr_values), min(ssim_values),
                         sum(ssim_values)/num_frames, max(ssim_values), min(lpips_values),
                         sum(lpips_values)/num_frames, max(lpips_values)])


if __name__ == '__main__':
    args = parser.parse_args()
    model_types = ['pytorch + generator_onnx', 'onnx', 'pytorch', 'pytorch + kp_onnx']
    if platform.system() == 'Linux':
        model_types.append('deepsparse')
    for model_type in model_types:
        run_inference(args.video_path, model_type, args.csv_file_name, args.config_path, args.onnx_path)
