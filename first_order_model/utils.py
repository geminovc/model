import yaml
import torch
import numpy as np
from first_order_model.modules.generator import OcclusionAwareGenerator
from first_order_model.modules.sr_generator import SuperResolutionGenerator
from first_order_model.modules.discriminator import MultiScaleDiscriminator
from first_order_model.modules.keypoint_detector import KPDetector
#from swinir_wrapper import SuperResolutionModel

def frame_to_tensor(frame, device):
    """ convert numpy arrays to tensors for reconstruction pipeline """
    array = np.expand_dims(frame, 0).transpose(0, 3, 1, 2)
    tensor = torch.from_numpy(array)
    return tensor.float().to(device)


def safe_read(config, first_key, second_key, default):
    try:
        return config[first_key][second_key]
    except Exception as e:
        print(e)
        return default

def get_main_config_params(config_path):
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except:
        config = None

    frame_shape = safe_read(config, 'dataset_params', 'frame_shape', [1024, 1024, 3])
    generator_params = safe_read(config, 'model_params', 'generator_params', {})
    use_lr_video = generator_params.get('use_lr_video', False)
    lr_size = generator_params.get('lr_size', 64)
    generator_type = generator_params.get('generator_type', 'occlusion_aware')

    return {'frame_shape': frame_shape,
            'use_lr_video': use_lr_video,
            'lr_size': lr_size,
            'generator_type': generator_type
            }


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
        6. SwinIR
    """
    generator_params = config['model_params']['generator_params']
    generator_type = generator_params.get('generator_type', 'occlusion_aware')
    if generator_type == 'swinir':
        generator = SuperResolutionModel(config)
        discriminator = None
    elif generator_type not in ['vpx', 'bicubic']:
        if generator_type in ['occlusion_aware', 'split_hf_lf']:
            generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        elif generator_type == 'just_upsampler':
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

