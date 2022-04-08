from first_order_model.fom_wrapper import FirstOrderModel
import imageio 
import numpy as np
import time
from argparse import ArgumentParser
from frames_dataset import get_num_frames, get_frame

parser = ArgumentParser()
parser.add_argument("--config", 
                    default="config/paper_configs/resolution512_with_hr_skip_connections.yaml",
                    help="path to config")
parser.add_argument("--video_path",
                    default="512_kayleigh_10_second_0_1.mp4",
                    help="path to the video")
parser.add_argument("--output_name",
                    default="prediction",
                    help="name of the output file to be saved")
parser.add_argument("--output_fps",
                    default=30,
                    help="fps of the final video")
parser.add_argument("--source_update_frequency",
                    default=1800,
                    help="source update frequency")
parser.set_defaults(verbose=False)
opt = parser.parse_args()

model = FirstOrderModel(opt.config)

video_name = opt.video_path
num_frames = get_num_frames(video_name)
print(num_frames)

source = get_frame(video_name, 0, ifnormalize=False)
source_kp, _= model.extract_keypoints(source)
model.update_source(0, source, source_kp)
old_source_index = 0
predictions = []
times = []

# warm-up
for _ in range(100):
    source_kp['source_index'] = 0
    _ = model.predict(source_kp)

for i in range(0, num_frames):
    print(i)
    if i % opt.source_update_frequency == 0:
        source = get_frame(video_name, i, ifnormalize=False)
        source_kp, _= model.extract_keypoints(source)
        model.update_source(len(model.source_frames), source, source_kp) 
    
    driving = get_frame(video_name, i, ifnormalize=False)
    target_kp, source_index = model.extract_keypoints(driving)
    target_kp['source_index'] = source_index
    if source_index == old_source_index:
        update_source = False
    else:
        update_source = True
        old_source_index = source_index
    start = time.perf_counter()
    predictions.append(model.predict(target_kp, update_source))
    times.append(time.perf_counter() - start)

print(f"Average prediction time per frame is {sum(times)/len(times)}s.")
imageio.mimsave(f'{opt.output_name}_freq{opt.source_update_frequency}.mp4', predictions, fps = int(opt.output_fps))
