from first_order_model.fom_wrapper import FirstOrderModel
import imageio 
import numpy as np
import time

video_name = "/video-conf/scratch/vibhaa/custom_dataset/test/vibhaa_smiling_modified.mp4"
video_array = np.array(imageio.mimread(video_name))

source = video_array[0, :, :, :]
model = FirstOrderModel("config/fom_original.yaml")
source_kp, _= model.extract_keypoints(source)
model.update_source(0, source, source_kp)
old_source_index = 0
predictions = []
times = []
source_update_frequency = 5

# warm-up
for _ in range(100):
    source_kp['source_index'] = 0
    _ = model.predict(source_kp)

for i in range(1, len(video_array) - 1):
    if i % source_update_frequency == 0:
        source = video_array[i, :, :, :]
        source_kp, _= model.extract_keypoints(source)
        model.update_source(len(model.source_frames), source, source_kp) 
    
    driving = video_array[i, :, :, :] 
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
imageio.mimsave('prediction.mp4', predictions)
