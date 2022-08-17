from first_order_model.fom_wrapper import FirstOrderModel
import imageio 
import numpy as np
import time
from argparse import ArgumentParser
from frames_dataset import get_num_frames, get_frame
import torch
import torch.nn.functional as F
import piq
from skimage import img_as_float32
from first_order_model.modules.model import Vgg19
from first_order_model.reconstruction import frame_to_tensor
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
                    default=1800, type=int,
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

def get_lr_array(input_array, lr_size, device):
    lr_array = frame_to_tensor(img_as_float32(input_array), device)
    lr_array = F.interpolate(lr_array, lr_size).data.cpu().numpy()
    lr_array = np.transpose(lr_array, [0, 2, 3, 1])[0]
    lr_array *= 255
    lr_array = lr_array.astype(np.uint8)
    return lr_array

model = FirstOrderModel(opt.config)
use_lr_video, lr_size = model.get_lr_video_info()
video_name = opt.video_path
source = np.random.rand(model.get_shape()[0], model.get_shape()[1], model.get_shape()[2])
source_kp, _= model.extract_keypoints(source)
model.update_source(0, source, source_kp)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if use_lr_video:
    source_lr = get_lr_array(source, lr_size, device)

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
    if use_lr_video:
        _ = model.predict_with_lr_video(source_lr)
    else:
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
    start = time.perf_counter()
    if use_lr_video:
        driving_lr = get_lr_array(driving, lr_size, device)
        prediction = model.predict_with_lr_video(driving_lr)
    else:
        target_kp, source_index = model.extract_keypoints(driving)
        target_kp['source_index'] = source_index
        prediction = model.predict(target_kp)

    predictions.append(prediction)
    times.append(time.perf_counter() - start)
    
    lpips_num, ssim, psnr = visual_metrics(driving, prediction)
    lpips_list.append(lpips_num)
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    i += 1
    if i % 100 == 0:
        print("Predicted ", i)


print(f"Average prediction time per frame is {sum(times)/len(times)}s.")
imageio.mimsave(f'{opt.output_name}_freq{opt.source_update_frequency}.mp4', predictions, fps = int(opt.output_fps))
print(np.mean(lpips_list), np.mean(ssim_list), np.mean(psnr_list))
