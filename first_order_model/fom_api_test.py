from first_order_model.fom_wrapper import FirstOrderModel
import imageio 
import numpy as np
import time
import torch
import lpips

def lpips_val(driving, prediction):
    """ compute lpips between driving and prediction """
    original = np.transpose(driving, [2, 0, 1])
    original_tensor = torch.unsqueeze(torch.from_numpy(original), 0)

    prediction = np.transpose(prediction, [2, 0, 1])
    prediction_tensor = torch.unsqueeze(torch.from_numpy(prediction), 0)
    
    if torch.cuda.is_available():
        original_tensor = original_tensor.cuda()
        prediction_tensor = prediction_tensor.cuda()
    lpips_val = loss_fn_vgg(original_tensor, prediction_tensor).data.cpu().numpy().flatten()[0]
    return lpips_val

video_name = "/video-conf/scratch/pantea/1024_short_clips_pantea/test/idPani_20_1.mp4"

model = FirstOrderModel("config/api_sample.yaml")
source = np.random.rand(model.get_shape()[0], model.get_shape()[1], model.get_shape()[2])
source_kp, _= model.extract_keypoints(source)
model.update_source(0, source, source_kp)

predictions = []
times = []
source_update_frequency = 10

lpips_list = []
loss_fn_vgg = lpips.LPIPS(net='vgg')
if torch.cuda.is_available():
    loss_fn_vgg = loss_fn_vgg.cuda()

# warm-up
for _ in range(100):
    source_kp['source_index'] = 0
    _ = model.predict(source_kp)
model.reset()

reader = imageio.get_reader(video_name, "ffmpeg")

i = 0
for frame in reader:
    if i % source_update_frequency == 0:
        source = frame 
        source_kp, _= model.extract_keypoints(source)
        model.update_source(len(model.source_frames), source, source_kp) 
    i += 1
    
    driving = frame #video_array[i, :, :, :] 
    target_kp, source_index = model.extract_keypoints(driving)
    target_kp['source_index'] = source_index

    start = time.perf_counter()
    prediction = model.predict(target_kp)
    predictions.append(prediction)
    times.append(time.perf_counter() - start)
    
    """ compute LPIPS """
    lpips_list.append(lpips_val(driving, prediction))

print(f"Average prediction time per frame is {sum(times)/len(times)}s.")    
imageio.mimsave('prediction.mp4', predictions)
print(np.mean(lpips_list))
