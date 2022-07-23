from tqdm import trange
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater
from frames_dataset import MetricsDataset
import lpips

from aiortc.codecs.vpx import Vp8Encoder, Vp8Decoder, vp8_depayload
from aiortc.jitterbuffer import JitterFrame

def get_frame_from_video_codec(av_frame, quantizer):
    """ go through the encoder/decoder pipeline to get a 
        representative decoded frame
    """
    # encode every frame as a keyframe with new encoder/decoder
    encoder, decoder = Vp8Encoder(), Vp8Decoder()
    payloads, timestamp = encoder.encode(av_frame, quantizer=quantizer)
    payload_data = [vp8_depayload(p) for p in payloads]
    jitter_frame = JitterFrame(data=b"".join(payload_data), timestamp=timestamp)
    decoded_frames = decoder.decode(jitter_frame)
    decoded_frame = decoded_frames[0].to_rgb().to_ndarray()
    return decoded_frame, sum([len(p) for p in payloads])


def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params'] 
    generator_params = config['model_params']['generator_params']
    generator_type = generator_params.get('generator_type', 'occlusion_aware')
    use_lr_video = generator_params.get('use_lr_video', False)
    lr_size = generator_params.get('lr_size', 64)
    
    if config['model_params']['discriminator_params'].get('conditional_gan', False):
        train_params['conditional_gan'] = True
        assert(train_params['skip_generator_loading'])

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    if kp_detector is not None:
        optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), 
                lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    else:
        optimizer_kp_detector = None

    if checkpoint is not None and generator_type in ["occlusion_aware", "split_hf_lf"]:
        if train_params.get('skip_generator_loading', False):
            # set optimizers and discriminator to None to avoid bogus values and to start training from scratch
            start_epoch = Logger.load_cpk(checkpoint, None, None, kp_detector,
                                      None, None, None, dense_motion_network=generator.dense_motion_network,
                                      generator_type=generator_type)
            start_epoch = 0
        elif generator_params.get('upsample_factor', 1) > 1:
            hr_skip_connections = generator_params.get('use_hr_skip_connections', False)
            run_at_256 = generator_params.get('run_at_256', True)
            start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      None, None, None, None, upsampling_enabled=True,
                                      use_lr_video=use_lr_video,
                                      hr_skip_connections=hr_skip_connections, run_at_256=run_at_256,
                                      generator_type=generator_type)
            start_epoch = 0
        elif train_params.get('train_everything_but_generator', False):
            run_at_256 = generator_params.get('run_at_256', True)
            start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, None,
                                      optimizer_generator, optimizer_discriminator, 
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector,
                                      dense_motion_network=None, run_at_256=run_at_256,
                                      generator_type=generator_type)
        else:
            start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector, 
                                      dense_motion_network=generator.dense_motion_network,
                                      generator_type=generator_type)

    elif checkpoint is not None and generator_type == "super_resolution":
            start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, None,
                                      None, optimizer_discriminator,
                                      None, dense_motion_network=None, 
                                      generator_type=generator_type)
            start_epoch = 0
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    if kp_detector is not None:
        scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    else:
        scheduler_kp_detector = None

    # train only generator parameters and keep dense motion/keypoint stuff frozen
    if train_params.get('train_only_generator', False) and checkpoint is not None:
        if kp_detector is not None:
            for param in kp_detector.parameters():
                param.requires_grad = False
            ev_loss = train_params['loss_weights']['equivariance_value']
            ev_jacobian = train_params['loss_weights']['equivariance_jacobian']
            assert ev_loss == 0 and ev_jacobian == 0, "Equivariance losses must be 0 to freeze keypoint detector"

            for param in generator.dense_motion_network.parameters():
                param.requires_grad = False
    elif train_params.get('train_everything_but_generator', False) and checkpoint is not None:
        for param in generator.parameters():
            param.requires_grad = False
        
        for param in generator.dense_motion_network.parameters():
            param.requires_grad = True

    # train only new layers added to increase resolution while keeping the rest of the pipeline frozen
    if train_params.get('train_only_non_fom_layers', False) and checkpoint is not None:
        if kp_detector is not None:
            for param in kp_detector.parameters():
                param.requires_grad = False
            ev_loss = train_params['loss_weights']['equivariance_value']
            ev_jacobian = train_params['loss_weights']['equivariance_jacobian']
            assert ev_loss == 0 and ev_jacobian == 0, "Equivariance losses must be 0 to freeze keypoint detector"

        for name, param in generator.named_parameters():
            if not(name.startswith("sigmoid") or name.startswith("hr") or name.startswith("final")):
                param.rquires_grad = False

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)
   
    metrics_dataloader = None
    if 'metrics_params' in config:
        metrics_dataset = MetricsDataset(**config['metrics_params'])
        metrics_dataloader = DataLoader(metrics_dataset, batch_size=train_params['batch_size'], shuffle=False, 
                num_workers=6, drop_last=False)

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)
    loss_fn_vgg = lpips.LPIPS(net='vgg')

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)
        loss_fn_vgg = loss_fn_vgg.cuda()

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                if use_lr_video:
                    lr_frame = F.interpolate(x['driving'], lr_size)
                    if train_params.get('encode_video_for_training', False):
                        quantizer = random.randint(0, 63)
                        x['driving_lr'] = get_frame_from_video_codec(lr_frame, quantizer) 
                    else:
                        x['driving_lr'] = lr_frame

                losses_generator, generated = generator_full(x, generator_type)

                if epoch == 0:
                    break

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()

                if optimizer_kp_detector is not None:
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            if epoch > 0:
                scheduler_generator.step()
                scheduler_discriminator.step()
                if scheduler_kp_detector is not None:
                    scheduler_kp_detector.step()
           
            # record a standard set of metrics
            if metrics_dataloader is not None:
                with torch.no_grad():
                    for i, y in enumerate(metrics_dataloader):
                        if use_lr_video:
                            lr_frame = F.interpolate(y['driving'], lr_size)
                            if train_params.get('encode_video_for_training', False):
                                quantizer = random.randint(0, 63)
                                y['driving_lr'] = get_frame_from_video_codec(lr_frame, quantizer) 
                            else:
                                y['driving_lr'] = lr_frame


                        _, metrics_generated = generator_full(y, generator_type)
                        logger.log_metrics_images(i, y, metrics_generated, loss_fn_vgg)

            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector}, inp=x, out=generated)
