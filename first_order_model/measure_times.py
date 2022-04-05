import torch
from torch import nn
from torch.autograd import grad
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import yaml
import csv
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
from first_order_model.modules.generator import OcclusionAwareGenerator


USE_QUANTIZATION = False
USE_CUDA = True
USE_FAST_CONV2 = False
NUM_RUNS = 1000
WARM_UP = 100


def get_mean(test_list):
    if len(test_list) != 0:
        return sum(test_list) / len(test_list)
    return 0

def get_variance(test_list, mean):
    if len(test_list) != 0:
        variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list)
        res = variance ** 0.5
        return res
    return 0


def print_average_and_std(test_list, name):
    if len(test_list) != 0:
        mean = get_mean(test_list)
        res = get_variance(test_list, mean)
        print(f"{name}:: mean={round(mean, 6)}, std={round(res / mean * 100, 6)}%")
        return mean, res
    return 0, 0


def display_times(times, module_name, USE_FAST_CONV2, USE_QUANTIZATION, USE_FLOAT_16, IMAGE_RESOLUTION):
    print(f"using custom conv:{USE_FAST_CONV2}, using quantization:{USE_QUANTIZATION}")
    print(f"using float16:{USE_FLOAT_16}, batch_size:{BATCH_SIZE}, resolution:{IMAGE_RESOLUTION}")
    for key in times.keys():
        print_average_and_std(times[key], key)

    if USE_FAST_CONV2:
        module_name += '_fast_conv'
    if USE_QUANTIZATION:
        module_name += '_int8'
    if USE_FLOAT_16:
        module_name += '_float16'
    if IMAGE_RESOLUTION:
        module_name += f'_res{IMAGE_RESOLUTION}'
    if BATCH_SIZE:
        module_name += f'_batchsize{BATCH_SIZE}'

    with open(module_name + '.csv', 'w', encoding='UTF8') as f:
        header = ['measurement']
        header += [key for key in times.keys()]
        writer = csv.writer(f)
        writer.writerow(header)
        mean_row = ['mean']
        mean_row += [get_mean(times[key]) for key in times.keys()]
        std_row = ['std%']
        std_row += [100 * get_variance(times[key], get_mean(times[key])) / (get_mean(times[key]) + 1e-8) for key in times.keys()]
        writer.writerow(mean_row)
        writer.writerow(std_row)


def get_random_inputs(model_name):
    x0 = torch.randn(BATCH_SIZE, 3, IMAGE_RESOLUTION, IMAGE_RESOLUTION, requires_grad=False)
    x1 = torch.randn(BATCH_SIZE, 10, 2, requires_grad=False)
    x2 = torch.randn(BATCH_SIZE, 10, 2, 2, requires_grad=False)
    x3 = torch.randn(BATCH_SIZE, 10, 2, requires_grad=False)
    x4 = torch.randn(BATCH_SIZE, 10, 2, 2, requires_grad=False)
    
    if USE_FLOAT_16:
        x0 = torch.randn(BATCH_SIZE, 3, IMAGE_RESOLUTION, IMAGE_RESOLUTION, requires_grad=False, dtype=torch.float16)
        x1 = torch.randn(BATCH_SIZE, 10, 2, requires_grad=False, dtype=torch.float16)
        x2 = torch.randn(BATCH_SIZE, 10, 2, 2, requires_grad=False, dtype=torch.float16)
        x3 = torch.randn(BATCH_SIZE, 10, 2, requires_grad=False, dtype=torch.float16)
        x4 = torch.randn(BATCH_SIZE, 10, 2, 2, requires_grad=False, dtype=torch.float16)
    
    if USE_CUDA:
        x0 = x0.cuda()
        x1 = x1.cuda()
        x2 = x2.cuda()
        x3 = x3.cuda()
        x4 = x4.cuda()

    if model_name != "kp_detector":
        return x0, x1, x2, x3, x4
    else:
        return x0, None, None, None, None


def time_generator(model):
    model.eval()
    if USE_QUANTIZATION:
        model = quantize_generator(model, enable_meausre=False)

    x0, x1, x2, x3, x4 = get_random_inputs("generator")
    if USE_FLOAT_16:
        model.half()
        model_inputs = [x0, {'value':x1}, {'value':x3}, False]
    else:
        model_inputs = [x0, {'value':x1, 'jacobian':x2}, {'value':x3, 'jacobian':x4}, False]

    if USE_CUDA:
        model.cuda()
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    # warm-up
    for _ in range(WARM_UP):
        _, _ = model(*model_inputs)
    
    # measuring
    total_times = []
    times_dict = { 'first_time': [], 'down_blocks_time': [],'dense_motion_time':[],'deform_input_time':[],
                  'bottleneck_time': [], 'up_blocks_time': [], 'final_time': []}
    #with torch.autograd.profiler.emit_nvtx():
    with torch.no_grad():
        for rep in range(NUM_RUNS):
            starter.record()
            _, time_dict =  model(*model_inputs)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            print(curr_time)
            total_times.append(curr_time)
            for key in time_dict.keys():
                times_dict[key].append(time_dict[key])
            if SLEEP_DUR > 0:
                time.sleep(SLEEP_DUR)
    times_dict['total_with_print'] = total_times
    display_times(times_dict, 'generator', USE_FAST_CONV2, USE_QUANTIZATION, USE_FLOAT_16, IMAGE_RESOLUTION)


parser = ArgumentParser()
parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
parser.add_argument("--resolution", default=1024, help="image resolution")
parser.add_argument("--batch_size", default=1, help="image batch_size")
parser.add_argument("--sleep_dur", default=0, help="sleep duration in seconds")
parser.add_argument("--float16", dest="float16", action="store_true", help="use float16")
parser.set_defaults(verbose=False)
opt = parser.parse_args()

with open(opt.config) as f:
    config = yaml.load(f)

IMAGE_RESOLUTION = int(opt.resolution)
BATCH_SIZE = int(opt.batch_size)
USE_FLOAT_16 = opt.float16
SLEEP_DUR = float(opt.sleep_dur)
generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],**config['model_params']['common_params'])
time_generator(generator)
