from abc import abstractmethod
import torch

class KeypointBasedFaceModels(torch.nn.Module):
    @abstractmethod
    def extract_keypoints(self, frame):
        """ extract keypoints from the provided RGB frame """
        pass

    @abstractmethod
    def predict(self, target_keypoints):
        """ predict and return the target RGB frame 
            from its keypoints 
        """
        pass

    @abstractmethod
    def predict_with_source(self, target_keypoints, source_frame, source_keypoints):
        """ predict and return the target RGB frame 
            from its keypoints using the passed in source rather
            than model state
        """
        pass

    @abstractmethod
    def predict_with_lr_video(self, target_64x64):
        """ predict and return the target RGB frame 
            from a low-res version of it. 
        """
        pass

    @abstractmethod
    def update_source(self, index, source_frame, source_keypoints):
        """ update the source frame and keypoints used by the model
            based on the RGB frame received as input
        """
        pass
