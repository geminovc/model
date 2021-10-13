import sys
sys.path.append('/Users/panteababaahmadi/Documents/GitHub/nets_implementation/original_bilayer')
from bilayer_wrapper import BilayerAPI
import numpy as np
from PIL import Image

config_path = '/Users/panteababaahmadi/Documents/GitHub/Bilayer_Checkpoints/runs/\
my_model_no_frozen_yaw_V9mbKUqFx0o/args.yaml'

model = BilayerAPI(config_path)

img_base_path = '/Users/panteababaahmadi/Downloads/Datasets/per_video_1_three_datasets\
/imgs/unseen_test/id00015/V9mbKUqFx0o/00268/'
source_img_path = img_base_path + '0.jpg'
target_img_path = img_base_path + '10.jpg'

source_frame = np.asarray(Image.open(source_img_path))
target_frame = np.asarray(Image.open(target_img_path))

source_poses = model.extract_keypoints(source_frame)
model.update_source(source_poses, source_frame)

target_poses = model.extract_keypoints(target_frame)

# Passing the Target Frame
predicted_target = model.predict(target_poses, target_frame)
predicted_target.save("10_pred_target_with_the_target_frame.png")

# Not Passing the Target Frame
predicted_target = model.predict(target_poses)
predicted_target.save("10_pred_target_without_the_target_frame.png")

target_img_path = img_base_path + '100.jpg'
target_frame = np.asarray(Image.open(target_img_path))
target_poses = model.extract_keypoints(target_frame)

# Passing the Target Frame
predicted_target = model.predict(target_poses, target_frame)
predicted_target.save("100_pred_target_with_the_target_frame.png")

# Not Passing the Target Frame
predicted_target = model.predict(target_poses)
predicted_target.save("100_pred_target_without_the_target_frame.png")