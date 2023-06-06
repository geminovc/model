import torch
import yaml
import numpy as np
from skimage import img_as_float32
from first_order_model.logger import Logger
from first_order_model.utils import configure_fom_modules, frame_to_tensor
import sys
sys.path.append("..")
from keypoint_based_face_models import KeypointBasedFaceModels

""" Implementation of abstract class KeypointBasedFaceModels
    that uses the first order model to predict the driving 
    frame given target keypoints
   

    Example usage (given a video file)
    =================================
    video = np.array(imageio.mimread(video_name))
    video_array = img_as_float32(video)
    video_array = video_array.transpose((0, 3, 1, 2))
    source = video_array[:1, :, :, :]
    target = video_array[1:2, :, :, :]
    
    model = FirstOrderModel("temp.yaml")
    source_kp, source_idx = model.extract_keypoints(source)
    model.update_source(0, source, source_kp)
    target_kp = model.extract_keypoints(target)
    prediction = model.predict(target_kp))
"""
class FirstOrderModel(KeypointBasedFaceModels):
    def __init__(self, config_path, checkpoint='None'):
        super(FirstOrderModel, self).__init__()
        
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # config parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator_params = config['model_params']['generator_params']
        self.shape = config['dataset_params']['frame_shape']
        self.use_lr_video = generator_params.get('use_lr_video', False)
        self.lr_size = generator_params.get('lr_size', 64)
        self.generator_type = generator_params.get('generator_type', 'occlusion_aware')

        # configure modules
        self.generator, _, self.kp_detector = configure_fom_modules(config, self.device)
        # initialize weights
        if checkpoint == 'None':
            checkpoint = config['checkpoint_params']['checkpoint_path']
        Logger.load_cpk(checkpoint, generator=self.generator, 
                kp_detector=self.kp_detector, device=self.device,
                dense_motion_network=self.generator.dense_motion_network)

        # set to test mode
        self.generator.eval()
        self.kp_detector.eval()
        
        # placeholders for source information
        self.source_keypoints = {}
        self.source_frames = {}
        self.last_source_index = -1
   
        timing_enabled = True
        self.times = []
        self.start = torch.cuda.Event(enable_timing=timing_enabled)
        self.end = torch.cuda.Event(enable_timing=timing_enabled)


    def get_shape(self):
        return tuple(self.shape)


    def get_lr_video_info(self):
        return self.use_lr_video, self.lr_size


    def reset(self):
        """ reset stats """
        self.times = []
        self.source_keypoints.clear()
        self.source_frames.clear()
        self.last_source_index = -1


    def update_source(self, index, source_frame, source_keypoints):
        """ update the source and keypoints the frame is using 
            from the RGB source provided as input
        """
        self.source_frames[index] = self.convert_image_to_tensor(source_frame)
        self.source_keypoints[index] = self.convert_kp_dict_to_tensors(source_keypoints)


    def extract_keypoints(self, frame):
        """ extract keypoints into a keypoint dictionary with/without jacobians
            from the provided RGB image
            uses keypoints to detect the best source image to use
        """
        if not torch.is_tensor(frame):
            transformed_frame = self.convert_image_to_tensor(frame)
        else:
            transformed_frame = frame
        
        # change to arrays and standardize
        # Note: keypoints are stored at key 'value' in FOM
        keypoint_struct = self.kp_detector(transformed_frame)
        keypoint_struct['value'] = keypoint_struct['value'].data.cpu().numpy()[0]
        keypoint_struct['keypoints'] = keypoint_struct.pop('value')
        if 'jacobian' in keypoint_struct:
            keypoint_struct['jacobian'] = keypoint_struct['jacobian'].data.cpu().numpy()[0]
            keypoint_struct['jacobians'] = keypoint_struct.pop('jacobian')
        
        return keypoint_struct, self.best_source_frame_index(frame, keypoint_struct)

    
    def best_source_frame_index(self, frame, keypoint_struct):
        """ return best source frame to use for prediction for these keypoints
            and update source frame list if need be
        """
        if len(self.source_frames) == 0:
            return 0
        return list(self.source_frames.keys())[-1]


    def convert_kp_dict_to_tensors(self, keypoint_dict):
        """ takes a keypoint dictionary and tensors the values appropriately """
        new_kp_dict = {}
        
        # Note: keypoints are stored at key 'value' in FOM
        new_kp_dict['value'] = torch.from_numpy(keypoint_dict['keypoints'])
        new_kp_dict['value'] = torch.unsqueeze(new_kp_dict['value'], 0)
        new_kp_dict['value'] = new_kp_dict['value'].float()

        if 'jacobians' in keypoint_dict:
            new_kp_dict['jacobian'] = torch.from_numpy(keypoint_dict['jacobians'])
            new_kp_dict['jacobian'] = torch.unsqueeze(new_kp_dict['jacobian'], 0)
            new_kp_dict['jacobian'] = new_kp_dict['jacobian'].float()
        
        if torch.cuda.is_available():
            for k in new_kp_dict.keys():
                new_kp_dict[k] = new_kp_dict[k].cuda() 

        return new_kp_dict


    def convert_image_to_tensor(self, image):
        """ takes an RGB image in 0-255 range and converts it
            into a float tensor in 0.0-1.0 range """
        transformed_image = frame_to_tensor(img_as_float32(image), self.device)

        return transformed_image


    def predict(self, target_keypoints, target_lr=None):
        """ takes target keypoints and returns an RGB image for the prediction """
        source_index = target_keypoints['source_index']
        if not torch.is_tensor(target_lr):
            target_lr_tensor = self.convert_image_to_tensor(target_lr)
        else:
            target_lr_tensor = target_lr

        assert(source_index in self.source_keypoints)
        assert(source_index in self.source_frames) 
        
        update_source = not (source_index == self.last_source_index)
        self.last_source_index = source_index
        
        source_kp_tensors = self.source_keypoints[source_index]
        target_kp_tensors = self.convert_kp_dict_to_tensors(target_keypoints)
        out = self.generator(self.source_frames[source_index], \
            kp_source=source_kp_tensors, kp_driving=target_kp_tensors, \
            update_source=update_source, driving_lr=target_lr_tensor)


        prediction = torch.mul(out['prediction'][0], 255).to(torch.uint8)
        prediction_cpu = prediction.data.cpu().numpy()
        final_prediction = np.transpose(prediction_cpu, [1, 2, 0])
        return final_prediction


    def predict_with_source(self, target_keypoints, source_frame, source_keypoints, target_lr=None):
        """ takes target keypoints and returns an RGB image for the prediction 
            using the source passed in
        """
        update_source = True
        self.last_source_index = -1 
        
        source_kp_tensors = self.convert_kp_dict_to_tensors(source_keypoints)
        target_kp_tensors = self.convert_kp_dict_to_tensors(target_keypoints)
        
        transformed_source = self.convert_image_to_tensor(source_frame)
        out = self.generator(transformed_source, \
                kp_source=source_kp_tensors, kp_driving=target_kp_tensors,\
                update_source=update_source, driving_lr=target_lr)

        prediction = torch.mul(out['prediction'][0], 255).to(torch.uint8)
        prediction_cpu = prediction.data.cpu().numpy()
        final_prediction = np.transpose(prediction_cpu, [1, 2, 0])
        return final_prediction


    def predict_with_lr_video(self, target_lr):
        """ predict and return the target RGB frame 
            from a low-res version of it. 
        """
        target_lr_tensor = self.convert_image_to_tensor(target_lr)
        target_keypoints, best_source_index = self.extract_keypoints(target_lr_tensor)
        target_keypoints['source_index'] = best_source_index

        return self.predict(target_keypoints, target_lr_tensor)


    def predict_with_lr_video_and_source(self, target_lr, source_frame, source_keypoints):
        """ predict and return the target RGB frame 
            from a low-res version of it. 
        """
        target_lr_tensor = self.convert_image_to_tensor(target_lr)
        target_keypoints, _ = self.extract_keypoints(target_lr_tensor)
        return self.predict_with_source(target_keypoints, source_frame, source_keypoints, target_lr_tensor)

