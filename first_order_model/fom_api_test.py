from first_order_model.fom_wrapper import FirstOrderModel
import imageio 
import numpy as np
import time

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

video_name = "/video-conf/scratch/vibhaa/custom_dataset/test/vibhaa_smiling_modified.mp4"
video_array = np.array(imageio.mimread(video_name))

source = video_array[0, :, :, :]
model = FirstOrderModel("config/api_sample.yaml")
print("Number of elements in the generator", get_n_params(model.generator))

source_kp = model.extract_keypoints(source)
model.update_source(source, source_kp)
predictions = []
tt = []
for i in range(1, len(video_array) - 1):
    print(i)
    target_kp = model.extract_keypoints(driving)
    start_time = time.time()
    predictions.append(model.predict(target_kp))
    tt.append(time.time() - start_time)

print("Average prediction time per frame", sum(tt)/len(tt))
imageio.mimsave('prediction.mp4', predictions)
