import torch
from torch import nn
from torch.autograd import grad
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
# from mmcv.ops.point_sample import bilinear_grid_sample
import yaml
import imageio 
import time
import os, sys
import numpy as np
from tqdm import trange
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
import collections
from first_order_model.sync_batchnorm import DataParallelWithCallback
from first_order_model.modules.discriminator import MultiScaleDiscriminator
from first_order_model.frames_dataset import FramesDataset, DatasetRepeater
from first_order_model.logger import Logger, Visualizer
from first_order_model.modules.model import DiscriminatorFullModel
from first_order_model.quantization.utils import quantize_model,  print_average_and_std, get_params, \
                                                    get_coder_modules_to_fuse, QUANT_ENGINE, display_times
from first_order_model.quantization.quantized_building_modules import SameBlock2d, DownBlock2d, UpBlock2d, \
                                                    ResBlock2d, USE_FAST_CONV2, Hourglass, Encoder, Decoder
from first_order_model.quantization.quantized_main_modules import OcclusionAwareGenerator, OcclusionAwareGenerator_with_time, \
                                                    DenseMotionNetwork, DenseMotionNetwork_with_time, \
                                                    FirstOrderModel, KPDetector, GeneratorFullModel


USE_FLOAT_16 = False
USE_QUANTIZATION = False
USE_CUDA = False
IMAGE_RESOLUTION = 256
NUM_RUNS = 1000


def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']
    if USE_QUANTIZATION:
        optimizer_generator = torch.optim.Adam(get_params(generator), lr=train_params['lr_generator'], betas=(0.5, 0.999))
        optimizer_kp_detector = torch.optim.Adam(get_params(kp_detector), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    else:
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
        optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            print("epoch", epoch)
            for x in dataloader:
                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)
                if USE_QUANTIZATION:
                    loss = torch.autograd.Variable(loss, requires_grad=True)
                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
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

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector}, inp=x, out=generated)


def get_random_inputs(model_name):
    x0 = torch.randn(1, 3, IMAGE_RESOLUTION, IMAGE_RESOLUTION, requires_grad=False)
    x1 = torch.randn(1, 10, 2, requires_grad=False)
    x2 = torch.randn(1, 10, 2, 2, requires_grad=False)
    x3 = torch.randn(1, 10, 2, requires_grad=False)
    x4 = torch.randn(1, 10, 2, 2, requires_grad=False)
    
    if USE_FLOAT_16:
        x0 = torch.randn(1, 3, IMAGE_RESOLUTION, IMAGE_RESOLUTION, requires_grad=False, dtype=torch.float16)
        x1 = torch.randn(1, 10, 2, requires_grad=False, dtype=torch.float16)
        x2 = torch.randn(1, 10, 2, 2, requires_grad=False, dtype=torch.float16)
        x3 = torch.randn(1, 10, 2, requires_grad=False, dtype=torch.float16)
        x4 = torch.randn(1, 10, 2, 2, requires_grad=False, dtype=torch.float16)
    
    if USE_CUDA:
        x0 = x0.to("cuda")
        x1 = x1.to("cuda")
        x2 = x2.to("cuda")
        x3 = x3.to("cuda")
        x4 = x4.to("cuda")

    if model_name != "kp_detector":
        return x0, x1, x2, x3, x4
    else:
        return x0, None, None, None, None


def quantize_generator(model_fp32=OcclusionAwareGenerator(3, 10, 64, 512, 2, 6, True,
     {'block_expansion': 64, 'max_features': 1024, 'num_blocks': 5, 'scale_factor': 0.25}, True, False),
     enable_meausre=True):

    modules_to_fuse = get_coder_modules_to_fuse(len(model_fp32.dense_motion_network.hourglass.encoder.down_blocks), 
                                                prefix='dense_motion_network.hourglass.encoder.down_blocks')
    modules_to_fuse += get_coder_modules_to_fuse(len(model_fp32.dense_motion_network.hourglass.decoder.up_blocks),
                                                prefix='dense_motion_network.hourglass.decoder.up_blocks')    
    modules_to_fuse += [['first.conv', 'first.norm', 'first.relu']]
    modules_to_fuse += get_coder_modules_to_fuse(len(model_fp32.down_blocks), prefix='down_blocks')
    modules_to_fuse += get_coder_modules_to_fuse(len(model_fp32.up_blocks), prefix='up_blocks')
    modules_to_fuse += [['bottleneck.r0.conv1', 'bottleneck.r0.norm1'], 
                        ['bottleneck.r0.conv2', 'bottleneck.r0.norm2', 'bottleneck.r0.relu'],
                        ['bottleneck.r1.conv1', 'bottleneck.r1.norm1'],
                        ['bottleneck.r1.conv2', 'bottleneck.r1.norm2', 'bottleneck.r1.relu'],
                        ['bottleneck.r2.conv1', 'bottleneck.r2.norm1'],
                        ['bottleneck.r2.conv2', 'bottleneck.r2.norm2', 'bottleneck.r2.relu'],
                        ['bottleneck.r3.conv1', 'bottleneck.r3.norm1'],
                        ['bottleneck.r3.conv2', 'bottleneck.r3.norm2', 'bottleneck.r3.relu'],
                        ['bottleneck.r4.conv1', 'bottleneck.r4.norm1'],
                        ['bottleneck.r4.conv2', 'bottleneck.r4.norm2', 'bottleneck.r4.relu'],
                        ['bottleneck.r5.conv1', 'bottleneck.r5.norm1'],
                        ['bottleneck.r5.conv2', 'bottleneck.r5.norm2', 'bottleneck.r5.relu']]
    
    x0, x1, x2, x3, x4 = get_random_inputs("generator")
    model_int8 = quantize_model(model_fp32, modules_to_fuse, x0, x1, x2, x3, x4, enable_meausre)

    return model_int8


def quantize_kp_detector(model_fp32=KPDetector(32, 10, 3, 1024, 5, 0.1, True, 0.25, False, 0, False), enable_meausre=True):
    modules_to_fuse = get_coder_modules_to_fuse(len(model_fp32.predictor.encoder.down_blocks), prefix='predictor.encoder.down_blocks')
    modules_to_fuse += get_coder_modules_to_fuse(len(model_fp32.predictor.decoder.up_blocks), prefix='predictor.decoder.up_blocks')
    x0, x1, x2, x3, x4 = get_random_inputs("kp_detector")
    model_int8 = quantize_model(model_fp32, modules_to_fuse, x0, enable_meausre=enable_meausre)
    return model_int8


def quantize_pipeline():
    model = FirstOrderModel("config/api_sample.yaml")
    model.generator = quantize_generator(model.generator, enable_meausre=False)
    model.kp_detector =  quantize_kp_detector(model.kp_detector, enable_meausre=False)
    
    video_name = "short_test_video.mp4"
    video_array = np.array(imageio.mimread(video_name))
    source = video_array[0, :, :, :]
    source_kp = model.extract_keypoints(source)
    model.update_source(source, source_kp)
    predictions = []
    tt = []
    for i in range(1, len(video_array) - 1):
        print(i)
        driving = video_array[i, :, :, :] 
        target_kp = model.extract_keypoints(driving)
        start_time = time.time()
        predictions.append(model.predict(target_kp))
        tt.append(time.time() - start_time)

    print_average_and_std(tt, "Average prediction time per frame")
    imageio.mimsave('quantized_prediction.mp4', predictions)


def quantize_enc(model_fp32=Encoder(64, 44, 5, 1024), enable_meausre=True):
    modules_to_fuse = get_coder_modules_to_fuse(len(model_fp32.down_blocks), prefix='down_blocks')
    model_int8 = quantize_model(model_fp32, modules_to_fuse, torch.randn(1, 44, 64, 64), enable_meausre=enable_meausre)
    return model_int8


def quantize_dec(model_fp32=Decoder(64, 44, 5, 1024), enable_meausre=True):
    modules_to_fuse = get_coder_modules_to_fuse(len(model_fp32.up_blocks), prefix='up_blocks')
    x0 = [torch.randn(1, 44, 64, 64), torch.randn(1, 128, 32, 32),
                  torch.randn(1, 256, 16, 16), torch.randn(1, 512, 8, 8),
                  torch.randn(1, 1024, 4, 4), torch.randn(1, 1024, 2, 2)]
    model_int8 = quantize_model(model_fp32, modules_to_fuse, x0, enable_meausre=enable_meausre)
    return model_int8


def quantize_hrglass(model_fp32=Hourglass(64, 44, 5, 1024), enable_meausre=True):
    modules_to_fuse = get_coder_modules_to_fuse(len(model_fp32.encoder.down_blocks), 
                                                prefix='encoder.down_blocks')
    modules_to_fuse += get_coder_modules_to_fuse(len(model_fp32.decoder.up_blocks), 
                                                prefix='decoder.up_blocks')
    model_int8 = quantize_model(model_fp32, modules_to_fuse, torch.randn(1, 44, 64, 64), enable_meausre=enable_meausre)
    return model_int8


def quantize_dense_motion(model_fp32=DenseMotionNetwork(64, 5, 1024, 10, 3, True, 0.25, 0.01, False), enable_meausre=True):
    modules_to_fuse = get_coder_modules_to_fuse(len(model_fp32.hourglass.encoder.down_blocks), prefix='hourglass.encoder.down_blocks')
    modules_to_fuse += get_coder_modules_to_fuse(len(model_fp32.hourglass.decoder.up_blocks), prefix='hourglass.decoder.up_blocks')
    x0, x1, x2, x3, x4 = get_random_inputs("dense_motion")
    model_int8 = quantize_model(model_fp32, modules_to_fuse, x0, x1, x2, x3, x4, enable_meausre=enable_meausre)
    return model_int8


def quantize_resblock(model_fp32=ResBlock2d(256, 3, 1), enable_meausre=True):
    modules_to_fuse = [['conv1', 'norm1', 'relu'], ['conv2', 'norm2']]
    model_int8 = quantize_model(model_fp32, modules_to_fuse, torch.randn(1, 256, 64, 64), enable_meausre=enable_meausre)
    return model_int8 


def qat_train_resblock(model=ResBlock2d(16, 3, 1)):
    input_fp32 = torch.randn(1, 16, 4, 4)
    output_fp32 = torch.randn(1, 16, 4, 4)
    model.eval()
    model = quantize_resblock(model, False)
    model.train()
    
    loss_fn = nn.L1Loss()
    params = []
    for name, mod in model.named_modules():                             
        if isinstance(mod, torch.nn.quantized.Conv2d):                              
            weight, bias = mod._weight_bias()
            params.append(weight)
            params.append(bias)                        
    
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(0, NUM_RUNS):
        print("epoch", epoch)
        running_loss = 0.0
        images = input_fp32
        labels = output_fp32
        images = torch.autograd.Variable(images.to(device), requires_grad=True)
        labels = torch.autograd.Variable(labels.to(device), requires_grad=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


def fine_grained_timing_generator(model=OcclusionAwareGenerator_with_time(3, 10, 64, 512, 2, 6, True,
     {'block_expansion': 64, 'max_features': 1024, 'num_blocks': 5, 'scale_factor': 0.25}, True, False)):

    model.eval()

    if USE_QUANTIZATION:
        model = quantize_generator(model, enable_meausre=False)
    
    if USE_FLOAT_16:
        model.half()

    x0, x1, x2, x3, x4 = get_random_inputs("generator")
    if USE_CUDA:
        model.to("cuda")
    
    first_times, down_blocks_1_times, down_blocks_2_times, dense_morion_times, deform_times,\
        bottleneck_times, up_blocks_1_times, up_blocks_2_times, final_times, sigmoid_times, \
        tt = [], [], [], [], [], [], [], [], [], [], []

    for i in range(0, NUM_RUNS):
        print(i)
        start_time = time.time()
        res, first_time, down_blocks_time, dense_morion_time, deform_time, \
        bottleneck_time, up_blocks_time, final_time, sigmoid_time = \
        model(x0, {'value':x1, 'jacobian':x2}, {'value':x3, 'jacobian':x4})
        tt.append(time.time() - start_time)
        first_times.append(first_time)
        down_blocks_1_times.append(down_blocks_time[0])
        down_blocks_2_times.append(down_blocks_time[1])
        dense_morion_times.append(dense_morion_time)
        deform_times.append(deform_time)
        bottleneck_times.append(bottleneck_time)
        up_blocks_1_times.append(up_blocks_time[0])
        up_blocks_2_times.append(up_blocks_time[1])
        final_times.append(final_time)
        sigmoid_times.append(sigmoid_time)

    times = {'first_times': first_times,
             'down_blocks_1_times': down_blocks_1_times,
             'down_blocks_2_times': down_blocks_2_times,
             'dense_morion_times': dense_morion_times,
             'deform_times': deform_times,
             'bottleneck_times': bottleneck_times,
             'up_blocks_1_times': up_blocks_1_times,
             'up_blocks_2_times': up_blocks_2_times,
             'final_times': final_times,
             'sigmoid_times': sigmoid_times,
             'total_with_print':tt}

    display_times(times, 'generator', USE_FAST_CONV2, USE_QUANTIZATION, USE_FLOAT_16, IMAGE_RESOLUTION)


def fine_grained_timing_dense_motion(model=DenseMotionNetwork_with_time(64, 5, 1024, 10, 3, True, 0.25, 0.01, False)):
    model.eval()

    if USE_QUANTIZATION:
        model = quantize_dense_motion(model, enable_meausre=False)
    
    if USE_FLOAT_16:
        model.half()

    x0, x1, x2, x3, x4 = get_random_inputs("generator")
    
    if USE_CUDA:
        model.to("cuda")

    heatmap_representation_times, sparse_motion_times, create_deformed_source_times,\
         hourglass_times, mask_times, softmax_times, deformation_times,\
          occlusion_times, tt = [], [], [], [], [], [], [], [], []
    for i in range(0, NUM_RUNS):
        print(i)
        start_time = time.time()
        res, heatmap_representation_time, sparse_motion_time, create_deformed_source_time,\
         hourglass_time, mask_time, softmax_time, deformation_time, occlusion_time\
          = model(x0, {'value':x1, 'jacobian':x2}, {'value':x3, 'jacobian':x4})
        heatmap_representation_times.append(heatmap_representation_time)
        sparse_motion_times.append(sparse_motion_time)
        create_deformed_source_times.append(create_deformed_source_time)
        hourglass_times.append(hourglass_time)
        mask_times.append(mask_time)
        softmax_times.append(softmax_time)
        deformation_times.append(deformation_time)
        occlusion_times.append(occlusion_time)
        tt.append(time.time() - start_time)
    
    times = {'heatmap_representation_times': heatmap_representation_times,
             'sparse_motion_times': sparse_motion_times,
             'create_deformed_source_times': create_deformed_source_times,
             'hourglass_times': hourglass_times,
             'mask_times': mask_times,
             'softmax_times': softmax_times,
             'deformation_times': deformation_times,
             'occlusion_times': occlusion_times,
             'total_with_print':tt}

    display_times(times, 'dense_motion', USE_FAST_CONV2, USE_QUANTIZATION, USE_FLOAT_16, IMAGE_RESOLUTION)


if __name__ == "__main__":
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate", "measurement"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--enable_timing", dest="enable_timing", action="store_true", help="Time the model")
    parser.add_argument("--q_aware", dest="q_aware", action="store_true", help="quantization-aware training enabled")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()

    if opt.mode == "measurement":
        fine_grained_timing_dense_motion()
        fine_grained_timing_generator()
    else:
        with open(opt.config) as f:
            config = yaml.load(f)

        if opt.checkpoint is not None:
            log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
        else:
            log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
            log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],**config['model_params']['common_params'])
        if opt.q_aware:
            generator = quantize_generator(generator)
            quantization_config = torch.quantization.get_default_qat_qconfig(QUANT_ENGINE)
            generator.qconfig = quantization_config
            torch.quantization.prepare_qat(generator, inplace=True)

        generator.train()
        if torch.cuda.is_available():
            generator.to(opt.device_ids[0])
        if opt.verbose:
            print(generator)

        discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                                **config['model_params']['common_params'])
        if torch.cuda.is_available():
            discriminator.to(opt.device_ids[0])
        if opt.verbose:
            print(discriminator)

        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                **config['model_params']['common_params'])
        if opt.q_aware:
            kp_detector = quantize_kp_detector(kp_detector)
            kp_detector.qconfig = quantization_config
            torch.quantization.prepare_qat(kp_detector, inplace=True)

        kp_detector.train()
        if torch.cuda.is_available():
            kp_detector.to(opt.device_ids[0])

        if opt.verbose:
            print(kp_detector)

        dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
            copy(opt.config, log_dir)

        if opt.mode == 'train':
            print("Training...")
            train(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, dataset, opt.device_ids)
        elif opt.mode == 'reconstruction':
            print("Reconstruction...")
            reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, dataset, opt.enable_timing)
        elif opt.mode == 'animate':
            print("Animate...")
            animate(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)

