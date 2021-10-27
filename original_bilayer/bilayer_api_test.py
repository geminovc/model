from bilayer_wrapper import BilayerModel
import numpy as np
from PIL import Image

config_path = '/video-conf/scratch/pantea_experiments_chunky/per_video_freezing_checkpoints\
/per_video/from_paper/runs/my_model_no_frozen_yaw_V9mbKUqFx0o/args.yaml'
if_save = False

model = BilayerModel(config_path)

img_base_path = '/video-conf/scratch/pantea/per_person_1_three_datasets\
/imgs/unseen_test/id00015/V9mbKUqFx0o/00268/'
source_img_path = img_base_path + '0.jpg'
target_img_path = img_base_path + '10.jpg'

# Bilayer Model Prediction
source_frame = np.asarray(Image.open(source_img_path))
target_frame = np.asarray(Image.open(target_img_path))
source_poses = model.extract_keypoints(source_frame)
model.update_source(source_frame, source_poses)
target_poses = model.extract_keypoints(target_frame)
predicted_target = model.predict(target_poses)

if if_save:
    predicted_target.save("10_pred_target.png")
