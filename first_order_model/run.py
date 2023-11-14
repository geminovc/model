import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset
from voxceleb2_dataset import Voxceleb2Dataset
from first_order_model.utils import configure_fom_modules
import torch

from train import train
from reconstruction import reconstruction
from animate import animate
from shrink_util import set_module, set_gen_module, set_keypoint_module
from train_with_distillation import train_distillation

if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate", "distill"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--experiment_name", default='vox-256-standard', help="experiment name to save logs")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--netadapt_checkpoint", default=None, help="path to netadapt checkpoint to override generator (and kp detector but it is frozen)")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--profile", dest="profile", action="store_true", help="Only profile model")
    parser.add_argument("--enable_timing", dest="enable_timing", action="store_true", help="Time the model")
    parser.add_argument("--save_visualizations_as_images", action="store_true", help="Save visuals as raw images for residual")
    parser.add_argument("--reference_frame_update_freq", type=int, help="how frequently to update reference frame")
    parser.add_argument("--person_id", dest="person_id", type=str, default=None, help="train on specific person")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.mode == "reconstruction":
        if opt.netadapt_checkpoint is None:
            log_dir = os.path.dirname(opt.checkpoint)
        else:
            log_dir = os.path.dirname(opt.netadapt_checkpoint)
        log_dir = os.path.join(log_dir, 'reconstruction' + '_' + opt.experiment_name)
                
    else:
        log_dir = os.path.join(opt.log_dir, opt.experiment_name)
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    generator, discriminator, kp_detector = configure_fom_modules(config, opt.device_ids[0])
    if opt.verbose:
        print(generator)
        print(discriminator)
        print(kp_detector)
    config['dataset_params']['person_id'] = opt.person_id
    is_train = opt.mode in ['train', 'distill']
    if 'metrics_params' in config:
        config['metrics_params']['person_id'] = opt.person_id
    if "voxceleb2" in config['dataset_params']['root_dir']:
        dataset = Voxceleb2Dataset(is_train=is_train, **config['dataset_params'])
    else:
        dataset = FramesDataset(is_train=is_train, **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if opt.mode != "reconstruction" and not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)
    elif opt.mode == "reconstruction":
        copy(opt.config, log_dir)

    image_shape = config['dataset_params']['frame_shape'][0]

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, discriminator, kp_detector, opt.checkpoint,  log_dir, dataset, opt.device_ids, image_shape, opt.netadapt_checkpoint)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, dataset, opt.enable_timing, 
                opt.save_visualizations_as_images, opt.experiment_name, opt.reference_frame_update_freq,
                       opt.profile, opt.netadapt_checkpoint)
    elif opt.mode == 'distill':
        print("Distilling...")
        train_distillation(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, dataset, opt.device_ids)
    elif opt.mode == 'animate':
        print("Animate...")
        animate(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
