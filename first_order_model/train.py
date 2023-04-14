from tqdm import trange
import torch
from shrink_util import *
import torch.nn.functional as F
from torch.utils.data import DataLoader
from first_order_model.modules.model import Vgg19, VggFace16

from logger import Logger
from first_order_model.modules.model import GeneratorFullModel, DiscriminatorFullModel

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback
from skimage import img_as_float32

from frames_dataset import DatasetRepeater
from frames_dataset import MetricsDataset
from fractions import Fraction
import lpips
import random
import av
import numpy as np

from first_order_model.utils import get_frame_from_video_codec 

def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params'] 
    generator_params = config['model_params']['generator_params']
    generator_type = generator_params.get('generator_type', 'occlusion_aware')
    use_lr_video = generator_params.get('use_lr_video', False) or generator_type == 'just_upsampler'
    lr_size = generator_params.get('lr_size', 64)

    lr_video_locations = []
    dense_motion_params = generator_params.get('dense_motion_params', {})
    if dense_motion_params.get('concatenate_lr_frame_to_hourglass_input', False) or \
            dense_motion_params.get('use_only_src_tgt_for_motion', False) :
        lr_video_locations.append('hourglass_input')
    if dense_motion_params.get('concatenate_lr_frame_to_hourglass_output', False):
        lr_video_locations.append('hourglass_output')
    if generator_params.get('use_3_pathways', False) or \
            generator_params.get('concat_lr_video_in_decoder', False) or \
            generator_type == 'just_upsampler':
        lr_video_locations.append('decoder')

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

    if dense_motion_params.get('use_RIFE', False):
        use_RIFE = True
    else:
        use_RIFE = False

    
    if checkpoint is not None and generator_type in ["occlusion_aware", "split_hf_lf"]:
        if train_params.get('fine_tune_entire_model', False):
            start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      None if use_RIFE else optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector, 
                                      dense_motion_network=generator.dense_motion_network,
                                      generator_type=generator_type)
        elif train_params.get('skip_generator_loading', False):
            # set optimizers and discriminator to None to avoid bogus values and to start training from scratch
            start_epoch = Logger.load_cpk(checkpoint, None, None, kp_detector,
                                      None, None, None, dense_motion_network=generator.dense_motion_network,
                                      generator_type=generator_type)
            start_epoch = 0
        elif generator_params.get('upsample_factor', 1) > 1 or use_lr_video:
            hr_skip_connections = generator_params.get('use_hr_skip_connections', False)
            run_at_256 = generator_params.get('run_at_256', True)
            start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      None, None, None, None, upsampling_enabled=True,
                                      use_lr_video=lr_video_locations,
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
                                      None if use_RIFE else optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector, 
                                      dense_motion_network=generator.dense_motion_network,
                                      generator_type=generator_type)
            if use_RIFE:
                start_epoch = 0

    elif checkpoint is not None and generator_type == "just_upsampler":
        if train_params.get('fine_tune_entire_model', False):
            start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, None,
                                      optimizer_generator, optimizer_discriminator,
                                      None, dense_motion_network=None, 
                                      generator_type=generator_type)

        else:
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
                num_workers=6, drop_last=True)

    # Force this value for distillation trials
    generator_type = 'occlusion_aware'

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    vgg_model = Vgg19()
    original_lpips = lpips.LPIPS(net='vgg')
    vgg_face_model = VggFace16()
    
    if torch.cuda.is_available():
        original_lpips = original_lpips.cuda()
        vgg_model = vgg_model.cuda()
        vgg_face_model = vgg_face_model.cuda()

    loss_fn_vgg = vgg_model.compute_loss
    face_lpips = vgg_face_model.compute_loss

    for x in dataloader:
        if use_lr_video or use_RIFE:
            lr_frame = F.interpolate(x['driving'], lr_size)
            if train_params.get('encode_video_for_training', False):
                x['driving_lr'] = get_encoded_frame(train_params, lr_frame, x)
            else:
                x['driving_lr'] = lr_frame

        # Manually move to cuda because it isn't automatically moved
        move_to_gpu(x)
        break

    # Get gen input once for macs/speed calculations in the future
    get_gen_input(generator_full,x)
    start = total_macs(generator)
    print("Start macs: ", start)

    # Set up the netadapt hyper parameters
    # Target: target macs, Current: current macs, reduce_amount: amoutn of macs to reduce each iteration
    if train_params.get('netadapt', False):
        prune_rate = train_params.get('shrink_rate', 0.02)
        reduce_amount = start * prune_rate
        current = start
        target = start * train_params.get('target_shrink', 0.25)

    # Force load a netadapt checkpoint shrunk_gen points to
    reload_gen = train_params.get('shrunk_gen', None)
    if reload_gen is not None:
        state_dict =torch.load(reload_gen)
        set_module(generator_full, state_dict)
        optimizer_generator = torch.optim.Adam(generator_full.generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
        current = total_macs(copy.deepcopy(generator_full.generator))
        print('reloaded params, new macs is', current)

    if train_params.get('netadapt', False):
        sort = train_params.get('netadapt_sort', False)
        steps_per_it = train_params.get('netadapt_steps_per_it', -1)
        with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
            epoch = 0
            is_first_round = True
            while current > target:
                epoch += 1
                print("Current macs is: ", current)
                print("Start macs was: ", start)
                print("Shrink ratio is: ", current/start)
                if not is_first_round:

                    # Run one iteration of netadapt

                    generator_full.generator, generator_full.kp_extractor, generator_full.discriminator = reduce_macs(
                        generator_full.generator, current - reduce_amount,
                        current, generator_full.kp_extractor, generator_full.discriminator, train_params,
                        dataloader, metrics_dataloader, generator_type,
                        lr_size, generator_full, sort, steps_per_it, device_ids)
                    current = total_macs(generator_full.generator)
                    reduce_amount = current * prune_rate
                    
                is_first_round = False

                # This code is the same metrics dataloading code as below
                # Logs metrics to tensorboard and checkpoints model.
                if metrics_dataloader is not None:
                    with torch.no_grad():
                        for i, y in enumerate(metrics_dataloader):
                            if use_lr_video or use_RIFE:
                                lr_frame = F.interpolate(y['driving'], lr_size)
                                if train_params.get('encode_video_for_training', False):
                                    y['driving_lr'] = get_encoded_frame(train_params, lr_frame, y)
                                else:
                                    y['driving_lr'] = lr_frame

                            move_to_gpu(y)
                            losses_generator, metrics_generated = generator_full(y, generator_type)
                            losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                            losses['macs'] = current
                            logger.log_iter(losses=losses)
                            logger.log_metrics_images(i, y, metrics_generated, loss_fn_vgg, original_lpips, face_lpips)

                    print("Generator took (ms):", get_generator_time(generator_full, y))
                    logger.log_epoch(epoch, {'generator': generator_full.generator,
                                             'discriminator': generator_full.discriminator,
                                             'kp_detector': generator_full.kp_extractor,
                                             'optimizer_generator': optimizer_generator,
                                             'optimizer_discriminator': optimizer_discriminator,
                                             'optimizer_kp_detector': optimizer_kp_detector}, inp=y, out=metrics_generated)

        # Remake the generator optimizer because generator has been copied
        optimizer_generator = torch.optim.Adam(generator_full.generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))

        # For now I've stuck a return here to ensure we don't train/overwrite checkpoints after running netadapt
        return

    # I move the data parallel stuff down because it breaks netadapt for some unknown reason. It's also the reasona I need to manually move inputs over to the gpu within netadapth
    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)
    


    print(f'Encoding using {train_params.get("codec", "vp8")}')
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in tqdm(dataloader):
                if use_lr_video or use_RIFE:
                    lr_frame = F.interpolate(x['driving'], lr_size)
                    if train_params.get('encode_video_for_training', False):
                        codec = train_params.get('codec', 'vp8')
                        target_bitrate = train_params.get('target_bitrate', None)
                        quantizer_level = train_params.get('quantizer_level', -1)
                        if target_bitrate == 'random':
                            target_bitrate = np.random.randint(15, 75) * 1000
                        nr = x.get('time_base_nr', torch.ones(lr_frame.size(dim=0), dtype=int))
                        dr = x.get('time_base_dr', 30000 * torch.ones(lr_frame.size(dim=0), dtype=int))
                        x['driving_lr'] = get_frame_from_video_codec(lr_frame, nr, \
                                dr, quantizer_level, target_bitrate, codec) 
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
                        if use_lr_video or use_RIFE:
                            lr_frame = F.interpolate(y['driving'], lr_size)
                            if train_params.get('encode_video_for_training', False):
                                target_bitrate = train_params.get('target_bitrate', None)
                                quantizer_level = train_params.get('quantizer_level', -1)
                                if target_bitrate == 'random':
                                    target_bitrate = np.random.randint(15, 75) * 1000
                                nr = y.get('time_base_nr', torch.ones(lr_frame.size(dim=0), dtype=int))
                                dr = y.get('time_base_dr', 30000 * torch.ones(lr_frame.size(dim=0), dtype=int))
                                y['driving_lr'] = get_frame_from_video_codec(lr_frame, nr, \
                                        dr, quantizer_level, target_bitrate) 
                            else:
                                y['driving_lr'] = lr_frame

                        _, metrics_generated = generator_full(y, generator_type)
                        logger.log_metrics_images(i, y, metrics_generated, loss_fn_vgg, original_lpips, face_lpips)

            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector}, inp=x, out=generated)
