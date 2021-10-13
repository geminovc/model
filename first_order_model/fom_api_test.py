from first_order_model.fom_wrapper import FirstOrderModel
import imageio 
import numpy as np

video_name = "/video-conf/scratch/vibhaa/custom_dataset/test/vibhaa_smiling_modified.mp4"
video_array = np.array(imageio.mimread(video_name))

source = video_array[0, :, :, :]
model = FirstOrderModel("config/api_sample.yaml")
source_kp = model.extract_keypoints(source)
model.update_source(source, source_kp)
predictions = []

for i in range(1, len(video_array) - 1):
    driving = video_array[i, :, :, :] 
    target_kp = model.extract_keypoints(driving)
    predictions.append(model.predict(target_kp))
    
imageio.mimsave('prediction.mp4', predictions)
