import yaml
import torch
import numpy as np
from first_order_model.modules.generator import OcclusionAwareGenerator
from first_order_model.modules.sr_generator import SuperResolutionGenerator
from first_order_model.modules.discriminator import MultiScaleDiscriminator
from first_order_model.modules.keypoint_detector import KPDetector
from swinir_wrapper import SuperResolutionModel
from torchprofile import profile_macs
import os

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


def get_model_macs(log_dir, generator, kp_detector, device):
    BATCH_SIZE = 1 # reconstruction

    source_image = torch.randn(BATCH_SIZE, 3, 512, 512, requires_grad=False, device=device)
    driving_lr = torch.randn(BATCH_SIZE, 3, 64, 64, requires_grad = False, device=device)
    update_source = True
    kp_val1 = torch.randn(BATCH_SIZE, 10, 2, requires_grad=False, device=device)
    kp_jac1 = torch.randn(BATCH_SIZE, 10, 2, 2, requires_grad=False, device=device)
    kp_val2 = torch.randn(BATCH_SIZE, 10, 2, requires_grad=False, device=device)
    kp_jac2 = torch.randn(BATCH_SIZE, 10, 2, 2, requires_grad=False, device=device)
    model_inputs = (source_image, 
                    {'value':kp_val1, 'jacobian':kp_jac1}, 
                    {'value':kp_val2, 'jacobian':kp_jac2}, 
                    update_source,
                    driving_lr)

    with open(os.path.join(log_dir, 'model_macs.txt'), 'wt') as model_file:
        kp_macs = profile_macs(kp_detector, source_image)
        print('{}: {:.4g} G'.format('kp_detector macs', kp_macs / 1e9))
        model_file.write('{}: {:.4g} G'.format('kp_detector macs', kp_macs / 1e9))
    
        generator_macs = profile_macs(generator, model_inputs)
        print('{}: {:.4g} G'.format('generator macs', generator_macs / 1e9))
        model_file.write('{}: {:.4g} G'.format('generator macs', generator_macs / 1e9))


def get_model_info(log_dir, kp_detector, generator):
    """ get model summary information for the passed-in keypoint detector and 
        generator in a text file in the log directory """
    
    with open(os.path.join(log_dir, 'model_summary.txt'), 'wt') as model_file:
        for model, name in zip([kp_detector, generator], ['kp', 'generator']):
            if model is not None:
                number_of_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) 
                total_number_of_parameters = sum(p.numel() for p in model.parameters())
  
                model_file.write('%s %s: %s\n' % (name, 'total_number_of_parameters',
                        str(total_number_of_parameters)))
                model_file.write('%s %s: %s\n' % (name, 'number_of_trainable_parameters',
                        str(number_of_trainable_parameters)))


