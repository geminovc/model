""" Train routine for distillation.

    Sets up teacher model with standard convolutions. Sets up
    an efficientnet based generator to mimic the teacher model.
    By default, losses on the efficientnet version are computed
    relative to the teacher model, rather than the ground truth.
    This can be changed in models.py. Has support for only training
    the decoder part in an efficient manner, and keeping the rest
    frozen from the teacher model.
"""

from tqdm import trange
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import lpips
import random
import numpy as np

from first_order_model.utils import configure_fom_modules, get_encoded_frame, get_model_macs
from first_order_model.modules.model import Vgg19, VggFace16
from first_order_model.modules.model import GeneratorFullModel, DiscriminatorFullModel
from first_order_model.sync_batchnorm import DataParallelWithCallback
from first_order_model.frames_dataset import DatasetRepeater
from first_order_model.frames_dataset import MetricsDataset
from first_order_model.logger import Logger

def setup_teacher_model(teacher_cpk, config, device_ids):
    """ set up the teacher model from which the distillation will occur """
    generator, _, kp_detector = configure_fom_modules(config, device_ids[0], teacher=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Logger.load_cpk(teacher_cpk, generator=generator, 
            kp_detector=kp_detector, device=device, 
            dense_motion_network=generator.dense_motion_network, 
            generator_type='occlusion_aware_generator', reconstruction=True)

    if generator is not None:
        if torch.cuda.is_available():
            generator = DataParallelWithCallback(generator)
        generator.eval()
    if kp_detector is not None:
        if torch.cuda.is_available():
            kp_detector = DataParallelWithCallback(kp_detector)
        kp_detector.eval()

    return generator, kp_detector


def train_distillation(config, generator, discriminator, kp_detector, teacher_checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params'] 
    teacher_generator, teacher_kp_detector = setup_teacher_model(teacher_checkpoint, config, device_ids)
    print('set up teacher')
    
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], 
                            shuffle=True, num_workers=6, drop_last=True)
    print('set up dataloader')
   
    metrics_dataloader = None
    if 'metrics_params' in config:
        metrics_dataset = MetricsDataset(**config['metrics_params'])
        metrics_dataloader = DataLoader(metrics_dataset, batch_size=train_params['batch_size'], shuffle=False, 
                num_workers=6, drop_last=True)
    print('set up metrics dataloader')

    # full models with data parallel callback
    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)
    
    # optimizers
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), 
                lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), 
                lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))

    # schedulers
    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=-1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=-1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=-1)
    
    # train only generator parameters and keep dense motion/keypoint stuff frozen
    gen_training = train_params.get('train_only_generator', False)
    decoder_training = train_params.get('train_only_decoder', False)
    if (gen_training or decoder_training) and teacher_checkpoint is not None:
        print('loading kp detector, and dense motion from checkpoint, and freezing during training')
        Logger.load_cpk(teacher_checkpoint, generator, None, kp_detector,
                        None, None, None, dense_motion_network=generator.dense_motion_network,
                        generator_type='occlusion_aware', 
                        skip_generator_loading=gen_training, keep_encoder=decoder_training)
        
        for param in kp_detector.parameters():
            param.requires_grad = False
        ev_loss = train_params['loss_weights']['equivariance_value']
        ev_jacobian = train_params['loss_weights']['equivariance_jacobian']
        assert ev_loss == 0 and ev_jacobian == 0, "Equivariance losses must be 0 to freeze keypoint detector"

        # Freezes dense motion from the teacher in the student
        for param in generator.dense_motion_network.parameters():
            param.requires_grad = False

        # Ensure that none of the encoder blocks for the HR or LR 
        # pathways are training in the student since the encoder is
        # used sparingly in the end-to-end pipeline.
        if decoder_training:
            for param in generator.hr_first.parameters(): 
                param.requires_grad = False
            for param in generator.lr_first.parameters():
                param.requires_grad = False

            for param in generator.hr_down_blocks.parameters():
                param.requires_grad = False
            
            for param in generator.down_blocks.parameters():
                param.requires_grad = False

    image_size = config['dataset_params']['frame_shape'][0]
    generator_params = config['model_params']['generator_params']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr_size = generator_params.get('lr_size', 64)
    get_model_macs(log_dir, generator, kp_detector, device, lr_size, image_size)
    
    # set up loss functions
    print('setting up loss functions')
    original_lpips = lpips.LPIPS(net='vgg')

    # set up full model
    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)
        original_lpips = original_lpips.cuda()
    
    # start training
    start_epoch = 0
    generator_params = config['model_params']['generator_params']
    lr_size = generator_params.get('lr_size', 64)
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], 
                checkpoint_freq=train_params['checkpoint_freq']) as logger:
        print('Training')
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                lr_frame = F.interpolate(x['driving'], lr_size)
                if train_params.get('encode_video_for_training', False):
                    x['driving_lr'] = get_encoded_frame(train_params, lr_frame, x)
                else:
                    x['driving_lr'] = lr_frame

                # move into its own black/class for teacher
                with torch.no_grad():
                    kp_source = teacher_kp_detector(x['source'])
                    kp_driving = teacher_kp_detector(x['driving_lr'])
                    teacher_out = teacher_generator(x['source'], kp_source=kp_source, \
                                kp_driving=kp_driving, update_source=True, driving_lr=x['driving_lr'])
                    x['teacher'] = teacher_out['prediction']
                    x['teacher_encoded_output'] = teacher_out['encoded_output'] 

                losses_generator, generated = generator_full(x, generator_type='occlusion_aware')

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
                        lr_frame = F.interpolate(y['driving'], lr_size)
                        if train_params.get('encode_video_for_training', False):
                            y['driving_lr'] = get_encoded_frame(train_params, lr_frame, y)
                        else:
                            y['driving_lr'] = lr_frame

                        # get teacher outpuut
                        kp_source = teacher_kp_detector(y['source'])
                        kp_driving = teacher_kp_detector(y['driving_lr'])
                        teacher_out = teacher_generator(y['source'], kp_source=kp_source, \
                                kp_driving=kp_driving, update_source=True, driving_lr=y['driving_lr'])
                        y['teacher'] = teacher_out['prediction']
                        y['teacher_encoded_output'] = teacher_out['encoded_output'] 

                        _, metrics_generated = generator_full(y, generator_type='occlusion_aware')
                        logger.log_metrics_images(i, y, metrics_generated, None, original_lpips, None)

            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector}, inp=x, out=generated)
