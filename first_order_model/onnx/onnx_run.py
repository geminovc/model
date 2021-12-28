from first_order_model.fom_wrapper import FirstOrderModel
import numpy as np
import onnxruntime
import torch
import time
import imageio
import argparse
import csv

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(description='Get video information')
parser.add_argument('--video_path',
                        type = str,
                        default = '../short_test_video.mp4',
                        help = 'path to the video')

parser.add_argument('--csv_file_name',
                       type = str,
                       required = True,
                       help = 'name of csv file to write results out to')



def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def run_inference(video_name, model_type='onnx', csv_file_name='timings.csv'):
    video_array = np.array(imageio.mimread(video_name))
    source = video_array[0, :, :, :]

    if model_type == 'pytorch':
        # set up model
        model = FirstOrderModel("../config/api_sample.yaml")
        # set up driving keypoints
        source_kp = model.extract_keypoints(source)
        model.update_source(source, source_kp)
    elif model_type == 'pytorch + kp_onnx':
        model = FirstOrderModel("../config/api_sample.yaml")
        ort_session_kp_extractor = onnxruntime.InferenceSession("fom_kp.onnx")
        # set up driving keypoints
        source_kp, source_jacobian = ort_session_kp_extractor.run(None, {'source': np.array(np.transpose(source[None, :], (0, 3, 1, 2)), dtype=np.float32)/255})
        model.update_source(source, {'keypoints': np.squeeze(source_kp), 'jacobians': source_jacobian})
    elif model_type == 'pytorch + generator_onnx':
        model = FirstOrderModel("../config/api_sample.yaml")
        ort_session_generator = onnxruntime.InferenceSession("fom_gen.onnx")
        # set up driving keypoints
        source_kp = model.extract_keypoints(source)
        model.update_source(source, source_kp)
    elif model_type == 'onnx':
        ort_session_generator = onnxruntime.InferenceSession("fom_gen.onnx")
        ort_session_kp_extractor = onnxruntime.InferenceSession("fom_kp.onnx")
        # set up driving keypoints
        source_kp, source_jacobian = ort_session_kp_extractor.run(None, {'source': np.array(np.transpose(source[None, :], (0, 3, 1, 2)), dtype=np.float32)/255})

    predictions = []
    kp_times = []
    gen_times = []

    for i in range(1, len(video_array) - 1):
        driving = video_array[i, :, :, :]

        if model_type == 'pytorch':
            start = time.time()
            target_kp = model.extract_keypoints(driving)
            kp_times.append(time.time() - start)
            start = time.time()
            frame_next = model.predict(target_kp)
            gen_times.append(time.time() - start)
            predictions.append(frame_next)
        elif model_type == 'pytorch + kp_onnx':
            start = time.time()
            target_kp, target_jacobian = ort_session_kp_extractor.run(None, {'source': np.array(np.transpose(driving[None, :], (0, 3, 1, 2)), dtype=np.float32)/255})
            kp_times.append(time.time() - start)
            start = time.time()
            frame_next = model.predict({'keypoints': np.squeeze(target_kp), 'jacobians': target_jacobian})
            gen_times.append(time.time() - start)
            predictions.append(frame_next)
        elif model_type == 'pytorch + generator_onnx':
            start = time.time()
            target_kp = model.extract_keypoints(driving)
            kp_times.append(time.time() - start)
            ort_inputs = {'source_image': np.array(np.transpose(source[None, :], (0, 3, 1, 2)), dtype=np.float32)/255,
                          'kp_driving_v': target_kp['keypoints'][None, :],
                          'kp_source_v': source_kp['keypoints'][None, :]}
            start = time.time()
            ort_outs = ort_session_generator.run(None, ort_inputs)
            gen_times.append(time.time() - start)
            predictions.append(np.transpose(ort_outs[0].squeeze(), (1, 2, 0)))

        elif model_type == 'onnx':
            start = time.time()
            target_kp, target_jacobian = ort_session_kp_extractor.run(None, {'source': np.array(np.transpose(driving[None, :], (0, 3, 1, 2)), dtype=np.float32)/255})
            kp_times.append(time.time() - start)
            ort_inputs = {'source_image': np.array(np.transpose(source[None, :], (0, 3, 1, 2)), dtype=np.float32)/255,
                          'kp_driving_v': target_kp,
                          'kp_source_v': source_kp}
            start = time.time()
            ort_outs = ort_session_generator.run(None, ort_inputs)
            gen_times.append(time.time() - start)
            predictions.append(np.transpose(ort_outs[0].squeeze(), (1, 2, 0)))

    imageio.mimsave(f'prediction_{model_type}.mp4', predictions)
    print(f'Average prediction time for {model_type}:', sum(gen_times)/len(gen_times))

    with open('model_type_' + csv_file_name, 'w', encoding='UTF8') as f:
        header = ['model', 'network','min_time', 'mean_time', 'max_time']
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow([model_type, 'generator',max(gen_times), sum(gen_times)/len(gen_times), min(gen_times)])
        writer.writerow([model_type, 'kp_detector',max(kp_times), sum(kp_times)/len(kp_times), min(kp_times)])


if __name__ == '__main__':
    args = parser.parse_args()

    for model_type in ['onnx', 'pytorch', 'pytorch + generator_onnx']:
        run_inference(args.video_path, model_type, args.csv_file_name)

