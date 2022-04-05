import torch
import yaml
import numpy as np
from skimage import img_as_float32
from first_order_model.logger import Logger
from first_order_model.modules.generator import OcclusionAwareGenerator
from first_order_model.modules.keypoint_detector import KPDetector

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
    def __init__(self, config_path):
        super(FirstOrderModel, self).__init__()
        
        with open(config_path) as f:
            config = yaml.safe_load(f)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        # generator
        self.generator = OcclusionAwareGenerator(
                **config['model_params']['generator_params'],
                **config['model_params']['common_params'])
        if torch.cuda.is_available():
            self.generator.to(device)

        # keypoint detector
        self.kp_detector = KPDetector(
                **config['model_params']['kp_detector_params'],
                **config['model_params']['common_params'])
        if torch.cuda.is_available():
            self.kp_detector.to(device)

        # initialize weights
        checkpoint = config['checkpoint_params']['checkpoint_path']
        Logger.load_cpk(checkpoint, generator=self.generator, 
                kp_detector=self.kp_detector, device=device)

        # set to test mode
        self.generator.eval()
        self.kp_detector.eval()
        
        # placeholders for source information
        self.source_keypoints = {}
        self.source_frames = {}
   
        timing_enabled = True
        self.times = []
        self.start = torch.cuda.Event(enable_timing=timing_enabled)
        self.end = torch.cuda.Event(enable_timing=timing_enabled)


    def reset(self):
        """ reset stats """
        self.times = []


    def update_source(self, index, source_frame, source_keypoints):
        """ update the source and keypoints the frame is using 
            from the RGB source provided as input
        """
        transformed_source = source_frame.transpose((2, 0, 1))
        transformed_source = torch.from_numpy(transformed_source).to(torch.uint8)
        if torch.cuda.is_available():
            transformed_source = transformed_source.cuda()

        # convert to float
        transformed_source = transformed_source.to(torch.float32)
        transformed_source = torch.div(transformed_source, 255)
        transformed_source = torch.unsqueeze(transformed_source, 0)
        self.source_frames[index] = transformed_source
        
        self.source_keypoints[index] = self.convert_kp_dict_to_tensors(source_keypoints)


    def extract_keypoints(self, frame):
        """ extract keypoints into a keypoint dictionary with/without jacobians
            from the provided RGB image
            uses keypoints to detect the best source image to use
        """
        transformed_frame = frame.transpose((2, 0, 1))
        transformed_frame = torch.from_numpy(transformed_frame).to(torch.uint8)
        if torch.cuda.is_available():
            transformed_frame = transformed_frame.cuda()
        
        # convert to float
        transformed_frame = transformed_frame.to(torch.float32)
        transformed_frame = torch.div(transformed_frame, 255)
        transformed_frame = torch.unsqueeze(transformed_frame, 0)
        
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
            return -1
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


    def predict(self, target_keypoints, update_source=False):
        """ takes target keypoints and returns an RGB image for the prediction """
        source_index = target_keypoints['source_index']
        assert(source_index in self.source_keypoints)
        assert(source_index in self.source_frames) 
	
        source_kp_tensors = self.source_keypoints[source_index]
        target_kp_tensors = self.convert_kp_dict_to_tensors(target_keypoints)
        #self.start.record()
        out = self.generator(self.source_frames[source_index], \
                kp_source=source_kp_tensors, kp_driving=target_kp_tensors, update_source=update_source)
        #self.end.record()
        #torch.cuda.synchronize()
        #last_val = self.start.elapsed_time(self.end)
        #self.times.append(last_val)
        #print(np.mean(self.times), len(self.times), last_val)

        prediction = torch.mul(out['prediction'][0], 255).to(torch.uint8)
        prediction_cpu = prediction.data.cpu().numpy()
        final_prediction = np.transpose(prediction_cpu, [1, 2, 0])
        return final_prediction
