import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset

from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector

import torch

from train import train
from reconstruction import reconstruction
from animate import animate

if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--experiment_name", default='vox-256-standard', help="experiment name to save logs")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--enable_timing", dest="enable_timing", action="store_true", help="Time the model")
    parser.add_argument("--save_visualizations_as_images", action="store_true", help="Save visuals as raw images for residual")
    parser.add_argument("--reference_frame_update_freq", type=int, help="how frequently to update reference frame")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.mode == "reconstruction":
        log_dir = os.path.dirname(opt.checkpoint)
    else:
        log_dir = os.path.join(opt.log_dir, opt.experiment_name)
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

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
        reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, dataset, opt.enable_timing, 
                opt.save_visualizations_as_images, opt.experiment_name, opt.reference_frame_update_freq)
    elif opt.mode == 'animate':
        print("Animate...")
        animate(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
