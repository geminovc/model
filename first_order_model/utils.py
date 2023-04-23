import yaml
from torch import nn
import torch
import numpy as np
from skimage import img_as_float32
from first_order_model.modules.generator import OcclusionAwareGenerator
from first_order_model.modules.sr_generator import SuperResolutionGenerator
from first_order_model.modules.discriminator import MultiScaleDiscriminator
from first_order_model.modules.keypoint_detector import KPDetector
#from swinir_wrapper import SuperResolutionModel
from torchprofile import profile_macs
from fractions import Fraction
from aiortc.codecs.vpx import Vp9Encoder, Vp9Decoder, Vp8Encoder, Vp8Decoder, vp8_depayload
from aiortc.jitterbuffer import JitterFrame
import os
import av


def get_frame_from_video_codec(frame_tensor, nr_list, dr_list, quantizer, bitrate, version="vp8"):
    """ go through the encoder/decoder pipeline to get a 
        representative decoded frame
    """
    # encode every frame as a keyframe with new encoder/decoder
    frame_data = frame_tensor.data.cpu().numpy()
    decoded_data = np.zeros(frame_data.shape, dtype=np.uint8) 
    
    frame_data = np.transpose(frame_data, [0, 2, 3, 1])
    frame_data *= 255
    frame_data = frame_data.astype(np.uint8)
    
    nr_list = nr_list.data.cpu().numpy()
    dr_list = dr_list.data.cpu().numpy()

    for i, (frame, nr, dr) in enumerate(zip(frame_data, nr_list, dr_list)):
        av_frame = av.VideoFrame.from_ndarray(frame)
        av_frame.pts = 0
        av_frame.time_base = Fraction(nr, dr)
        
        if version == "vp8":
            encoder, decoder = Vp8Encoder(), Vp8Decoder()
        else:
            encoder, decoder = Vp9Encoder(), Vp9Decoder()
        
        if bitrate == None:
            payloads, timestamp = encoder.encode(av_frame, quantizer=quantizer, enable_gcc=False)
        else:
            payloads, timestamp = encoder.encode(av_frame, quantizer=-1, \
                    target_bitrate=bitrate, enable_gcc=False)
        payload_data = [vp8_depayload(p) for p in payloads]
        
        jitter_frame = JitterFrame(data=b"".join(payload_data), timestamp=timestamp)
        decoded_frames = decoder.decode(jitter_frame)
        decoded_frame = decoded_frames[0].to_rgb().to_ndarray()
        decoded_data[i] = np.transpose(decoded_frame, [2, 0, 1]).astype(np.uint8)
    return torch.from_numpy(img_as_float32(decoded_data))


def get_encoded_frame(train_params, lr_frame, data):
    """ extract target bitrate info, and nr and dr for time base """
    target_bitrate = train_params.get('target_bitrate', None)
    quantizer_level = train_params.get('quantizer_level', -1)
    if target_bitrate == 'random':
        target_bitrate = np.random.randint(15, 75) * 1000
    nr = data.get('time_base_nr', torch.ones(lr_frame.size(dim=0), dtype=int))
    dr = data.get('time_base_dr', 30000 * torch.ones(lr_frame.size(dim=0), dtype=int))

    return get_frame_from_video_codec(lr_frame, nr, dr, quantizer_level, target_bitrate) 


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


def configure_fom_modules(config, device, teacher=False):
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
    if teacher:
        generator_type = 'occlusion_aware'
        config['model_params']['generator_params']['generator_type'] = 'occlusion_aware'
    if generator_type == 'swinir':
        generator = SuperResolutionModel(config)
        discriminator = None
    elif generator_type not in ['vpx', 'bicubic']:
        if generator_type in ['occlusion_aware', 'split_hf_lf', 'student_occlusion_aware']:
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

    if generator_type in ['occlusion_aware', 'split_hf_lf', 'student_occlusion_aware']:
        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
        if torch.cuda.is_available():
            kp_detector.to(device)

    else:
        kp_detector = None
    
    return generator, discriminator, kp_detector

def get_input_features(module):

    all_layers = [
        m for n, m in module.named_modules()
        if (isinstance(m, nn.Conv2d)
            or isinstance(m, nn.modules.batchnorm._BatchNorm))
    ]
    return all_layers[0].in_channels

def get_model_macs(log_dir, generator, kp_detector, device, lr_size, image_size, BATCH_SIZE=1):
    # reconstruction

    source_image = torch.randn(BATCH_SIZE, 3, image_size, image_size, requires_grad=False, device=device)
    driving_lr = torch.randn(BATCH_SIZE, 3, lr_size, lr_size, requires_grad = False, device=device)
    update_source = True
    kp_val1 = torch.randn(BATCH_SIZE, 10, 2, requires_grad=False, device=device)
    kp_jac1 = torch.randn(BATCH_SIZE, 10, 2, 2, requires_grad=False, device=device)
    kp_val2 = torch.randn(BATCH_SIZE, 10, 2, requires_grad=False, device=device)
    kp_jac2 = torch.randn(BATCH_SIZE, 10, 2, 2, requires_grad=False, device=device)
    num_features = 256 if 'distillation' in log_dir else 512
    bottleneck_num_features  = get_input_features(generator.bottleneck)
    bottleneck_inp = torch.randn(BATCH_SIZE, bottleneck_num_features, 64, 64, requires_grad=False, device=device)

    model_inputs = (source_image, 
                    {'value':kp_val1, 'jacobian':kp_jac1}, 
                    {'value':kp_val2, 'jacobian':kp_jac2}, 
                    update_source,
                    driving_lr)

    dense_motion_inputs = (source_image, 
                    {'value':kp_val1, 'jacobian':kp_jac1}, 
                    {'value':kp_val2, 'jacobian':kp_jac2}, 
                    driving_lr)

    with open(os.path.join(log_dir, 'model_macs.txt'), 'wt') as model_file:
        kp_macs = profile_macs(kp_detector, source_image)
        print('{}: {:.4g} G'.format('kp_detector macs', kp_macs / 1e9))
        model_file.write('{}: {:.4g} G\n'.format('kp_detector macs', kp_macs / 1e9))
    
        generator_macs = profile_macs(generator, model_inputs)
        print('{}: {:.4g} G'.format('generator macs', generator_macs / 1e9))
        model_file.write('{}: {:.4g} G\n'.format('generator macs', generator_macs / 1e9))
        
        dense_motion_macs = profile_macs(generator.dense_motion_network, dense_motion_inputs)
        print('{}: {:.4g} G'.format('dense motion macs', dense_motion_macs / 1e9))
        model_file.write('{}: {:.4g} G\n'.format('dense motion macs', dense_motion_macs / 1e9))

        bottleneck_macs = profile_macs(generator.bottleneck, bottleneck_inp)
        print('{}: {:.4g} G'.format('bottleneck macs', bottleneck_macs / 1e9))
        model_file.write('{}: {:.4g} G\n'.format('bottleneck macs', bottleneck_macs / 1e9))

        encoder_macs = 0
        for i, b in enumerate(generator.hr_down_blocks + generator.down_blocks):
            dim = int(image_size/2**i)
            start = int(16 * (1024 / image_size))
            features = get_input_features(b)
            random_input =  torch.randn(BATCH_SIZE, features, dim, dim, requires_grad=False, device=device)
            encoder_macs += profile_macs(b, random_input)
        print('{}: {:.4g} G'.format('encoder macs', encoder_macs / 1e9))
        model_file.write('{}: {:.4g} G\n'.format('encoder macs', encoder_macs / 1e9))

        if 'distillation' not in log_dir:
            start_dim = 64
            decoder_macs = 0
            for i, b in enumerate(generator.up_blocks + generator.hr_up_blocks):
                dim = int(start_dim * 2**i)
                features = int(num_features / 2**i)
                if i >= len(generator.up_blocks):
                    features *= 2
                if dim == lr_size:
                    features += 32
                features = get_input_features(b)
                random_input =  torch.randn(BATCH_SIZE, features, dim, dim, requires_grad=False, device=device)
                decoder_macs += profile_macs(b, random_input)
            print('{}: {:.4g} G'.format('decoder macs', decoder_macs / 1e9))
            model_file.write('{}: {:.4g} G\n'.format('decoder macs', decoder_macs / 1e9))   
        else:
            skip_connections = []
            dim = int(image_size)
            features = int(16 * (1024 / image_size))
            while dim >= 256:
                #skip_features = get_input_features(generator.efficientnet_decoder._blocks[5])
                skip_connections.append(torch.randn(BATCH_SIZE, features, dim, dim, requires_grad=False, device=device))
                features *= 2
                dim = dim // 2
            lr_input = torch.randn(BATCH_SIZE, 32, lr_size, lr_size, requires_grad=False, device=device)
            decoder_macs = profile_macs(generator.efficientnet_decoder, (bottleneck_inp, lr_input, skip_connections))
            print('{}: {:.4g} G'.format('decoder macs', decoder_macs / 1e9))
            model_file.write('{}: {:.4g} G\n'.format('decoder macs', decoder_macs / 1e9))   

    return decoder_macs + bottleneck_macs


def get_model_info(log_dir, kp_detector, generator):
    """ get model summary information for the passed-in keypoint detector and 
        generator in a text file in the log directory """
    
    with open(os.path.join(log_dir, 'model_summary.txt'), 'wt') as model_file:
        for model, name in zip([kp_detector, generator], ['kp', 'generator']):
            if model is not None:
                number_of_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) 
                total_number_of_parameters = sum(p.numel() for p in model.parameters())

                if name == 'generator':
                    total_decoder_bottleneck_params = sum(p.numel() for p in model.bottleneck.parameters())
                    total_decoder_bottleneck_params += sum(p.numel() for p in model.up_blocks.parameters())
                    total_decoder_bottleneck_params += sum(p.numel() for p in model.hr_up_blocks.parameters())
                    model_file.write('%s %s: %s\n' % (name, 'total_number_of_decoder_bottleneck_parameters',
                            str(total_decoder_bottleneck_params)))

  
                model_file.write('%s %s: %s\n' % (name, 'total_number_of_parameters',
                        str(total_number_of_parameters)))
                model_file.write('%s %s: %s\n' % (name, 'number_of_trainable_parameters',
                        str(number_of_trainable_parameters)))


