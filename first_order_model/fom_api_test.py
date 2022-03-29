from first_order_model.fom_wrapper import FirstOrderModel
import imageio 
import numpy as np
import time

video_name = "/video-conf/scratch/vibhaa/custom_dataset/test/vibhaa_smiling_modified.mp4"
video_array = np.array(imageio.mimread(video_name))

source = video_array[0, :, :, :]
model = FirstOrderModel("config/api_sample.yaml")
source_kp, _= model.extract_keypoints(source)
model.update_source(0, source, source_kp)
predictions = []
times = []

for i in range(1, len(video_array) - 1):
    driving = video_array[i, :, :, :] 
    target_kp, _ = model.extract_keypoints(driving)
    start = time.time()
    predictions.append(model.predict(target_kp))
    times.append(time.time() - start)

print(f"Average prediction time per frame is {sum(times)/len(times)}s.")    
imageio.mimsave('prediction.mp4', predictions)
