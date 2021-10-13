import torch
import yaml
import numpy as np
from skimage import img_as_float32
from first_order_model.logger import Logger
from first_order_model.modules.generator import OcclusionAwareGenerator
from first_order_model.modules.keypoint_detector import KPDetector
from ..keypoint_based_face_models import KeypointBasedFaceModels

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
    model.update_source(source)
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
                kp_detector=self.kp_detector)

        # set to test mode
        self.generator.eval()
        self.kp_detector.eval()
        
        # placeholders for source information
        self.source_keypoints = None
        self.source = None


    def update_source(self, source_frame, source_keypoints):
        """ update the source and keypoints the frame is using 
            from the RGB source provided as input
        """
        transformed_source = np.array([img_as_float32(source_frame)])
        transformed_source = transformed_source.transpose((0, 3, 1, 2))
        self.source = torch.from_numpy(transformed_source)
        self.source_keypoints = source_keypoints 


    def extract_keypoints(self, frame):
        """ extract keypoint from the provided RGB image """
        transformed_frame = np.array([img_as_float32(frame)])
        transformed_frame = transformed_frame.transpose((0, 3, 1, 2))
        frame = torch.from_numpy(transformed_frame)
        if torch.cuda.is_available():
            frame = frame.cuda() 
        return self.kp_detector(frame)


    def predict(self, target_keypoints):
        """ takes target keypoints and returns an RGB image for the prediction """
        assert(self.source_keypoints is not None)
        assert(self.source is not None)

        if torch.cuda.is_available():
            self.source = self.source.cuda() 

        out = self.generator(self.source, \
                kp_source=self.source_keypoints, kp_driving=target_keypoints)
        prediction_cpu = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction_cpu, [0, 2, 3, 1])[0]
        return (255 * prediction).astype(np.uint8)

