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
timing_enabled = True


parser = ArgumentParser()
parser.add_argument("--config", 
                    default="config/paper_configs/resolution512_with_hr_skip_connections.yaml",
                    help="path to config")
parser.add_argument("--video-path",
                    default="512_kayleigh_10_second_0_1.mp4",
                    help="path to the video")
parser.add_argument("--output-name",
                    default="prediction",
                    help="name of the output file to be saved")
parser.add_argument("--output-fps",
                    default=30,
                    help="fps of the final video")
parser.add_argument("--source-update-frequency",
                    default=1800,
                    help="source update frequency")
parser.set_defaults(verbose=False)
opt = parser.parse_args()

def visual_metrics(driving, prediction):
    start = torch.cuda.Event(enable_timing=timing_enabled)
    start.record()
    driving = np.expand_dims(img_as_float32(driving), 0)
    prediction = np.expand_dims(img_as_float32(prediction), 0)
    
    driving = np.transpose(driving, [0, 3, 1, 2])
    prediction = np.transpose(prediction, [0, 3, 1, 2])
    
    original_tensor = torch.from_numpy(driving)
    prediction_tensor = torch.from_numpy(prediction)
    if torch.cuda.is_available():
        original_tensor = original_tensor.cuda()
        prediction_tensor = prediction_tensor.cuda()

    lpips_val = vgg_model.compute_loss(original_tensor, prediction_tensor).data.cpu().numpy().flatten()[0]
    ssim = piq.ssim(original_tensor, prediction_tensor, data_range=1.).data.cpu().numpy().flatten()[0]
    psnr = piq.psnr(original_tensor, prediction_tensor, data_range=1., \
            reduction='none').data.cpu().numpy().flatten()[0]

    end = torch.cuda.Event(enable_timing=timing_enabled)
    end.record()
    torch.cuda.synchronize()
    curr_time = start.elapsed_time(end)
    return lpips_val, ssim, psnr

model = FirstOrderModel(opt.config)
video_name = opt.video_path
source = np.random.rand(model.get_shape()[0], model.get_shape()[1], model.get_shape()[2])
source_kp, _= model.extract_keypoints(source)
model.update_source(0, source, source_kp)

predictions = []
times = []
lpips_list = []
psnr_list = []
ssim_list = []

vgg_model = Vgg19()
if torch.cuda.is_available():
    vgg_model = vgg_model.cuda()

# warm-up
for _ in range(100):
    source_kp['source_index'] = 0
    _ = model.predict(source_kp)
model.reset()

reader = imageio.get_reader(video_name, "ffmpeg")

i = 0
for frame in reader:
    if i % opt.source_update_frequency == 0:
        source = frame 
        source_kp, _= model.extract_keypoints(source)
        model.update_source(len(model.source_frames), source, source_kp) 

    driving = frame 
    target_kp, source_index = model.extract_keypoints(driving)
    target_kp['source_index'] = source_index

    start = time.perf_counter()
    prediction = model.predict(target_kp)
    predictions.append(prediction)
    times.append(time.perf_counter() - start)
    
    lpips_num, ssim, psnr = visual_metrics(driving, prediction)
    lpips_list.append(lpips_num)
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    i += 1


print(f"Average prediction time per frame is {sum(times)/len(times)}s.")
imageio.mimsave(f'{opt.output_name}_freq{opt.source_update_frequency}.mp4', predictions, fps = int(opt.output_fps))
print(np.mean(lpips_list), np.mean(ssim_list), np.mean(psnr_list))
