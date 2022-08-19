import torch
import numpy as np
from first_order_model.modules.generator import OcclusionAwareGenerator
from first_order_model.modules.sr_generator import SuperResolutionGenerator
from first_order_model.modules.discriminator import MultiScaleDiscriminator
from first_order_model.modules.keypoint_detector import KPDetector


def frame_to_tensor(frame, device):
    """ convert numpy arrays to tensors for reconstruction pipeline """
    array = np.expand_dims(frame, 0).transpose(0, 3, 1, 2)
    array = torch.from_numpy(array)
    return array.float().to(device)


def configure_fom_modules(config, device):
    """ Generator can be of the following types:
        1. VPX: (not model based) just runs through the regular VPX 
           decoder to decode the frame at its highest resolution
        2. Bicubic - does a simple bicubic-based upsampling from low-resolution
           to high-resolution frames
        3. Super-resolution: does a simple super-resolution using upsampling
           learnt blocks to generate the high-resolution image
        4. OcclusionAware: uses the standard FOM model with/without an additional
           low-resolution video in the decoder/hourglass to produce the high-resolution
           warped image in the desired orientation from a reference frame
        5. Split HF/LF: Generator that uses the Occlusion aware pipeline for
           High-frequency (HF) content and simple super-resolution for the 
           low-frequency (LF) content
    """
    generator_params = config['model_params']['generator_params']
    generator_type = generator_params.get('generator_type', 'occlusion_aware')
    if generator_type not in ['vpx', 'bicubic']:
        if generator_type in ['occlusion_aware', 'split_hf_lf']:
            generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        elif generator_type == 'super_resolution':
            generator = SuperResolutionGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        if torch.cuda.is_available():
            generator.to(device)

        discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                                **config['model_params']['common_params'])
        if torch.cuda.is_available():
            discriminator.to(device)

    else: # VPX/Bicubic
        generator = None
        discriminator = None

    if generator_type in ['occlusion_aware', 'split_hf_lf']:
        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
        if torch.cuda.is_available():
            kp_detector.to(device)

    else:
        kp_detector = None
    
    return generator, discriminator, kp_detector

