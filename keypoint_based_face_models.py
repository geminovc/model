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
    def update_source(self, source_frame, source_keypoints):
        """ update the source frame and keypoints used by the model
            based on the RGB frame received as input
        """
        pass
