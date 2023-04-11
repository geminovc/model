from tqdm import trange
import gc
import os
from tqdm import tqdm
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import torch
import torch.nn.functional as F
import first_order_model
from torchprofile import profile_macs
import torch_pruning as tp

import sys
from torch.utils.data import DataLoader
from first_order_model.modules.model import Vgg19, VggFace16
import torch.nn.utils.prune as prune

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback
from skimage import img_as_float32
import copy
from frames_dataset import DatasetRepeater
from frames_dataset import MetricsDataset
from fractions import Fraction
import lpips
import random
import av
import numpy as np
import torch.nn as nn

from aiortc.codecs.vpx import Vp8Encoder, Vp8Decoder, vp8_depayload
from aiortc.jitterbuffer import JitterFrame
from matplotlib import pyplot as plt
import matplotlib
import warnings


def get_attr(obj, names):
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])

def get_attr_default(obj, names, default):
    if len(names) == 1:
        return getattr(obj, names[0], default)
    else:
        return get_attr_default(getattr(obj, names[0], default), names[1:], default)

def set_attr(obj, names, val):
    if len(names) == 0:
        with torch.no_grad():
            obj.set_(val)
        #setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def print_gen_module(state_dict):
    for key1 in state_dict.keys():
        if key1 not in ['generator', 'kp_detector']:
            continue
        for key, dict_param in state_dict[key1].items():
            if key1 == 'kp_detector':
                submod_names = ['kp_extractor'] + key.split(".")
            else:
                submod_names = [key1] + key.split(".")
            #curr_param = get_attr(mod, submod_names)
            # Here you can either replace the existing one

            if 'norm' not in key and 'bias' not in key:
                print(dict_param.shape, key)
            #set_attr(mod, submod_names, dict_param)

def print_diff(state_dict, state_dict2):
    for key1 in state_dict.keys():
        if key1 not in ['generator']:
            continue
        for key, dict_param in state_dict[key1].items():
            if 'norm' not in key and 'bias' not in key:
                if state_dict2[key1][key].shape != dict_param.shape:

                    print(dict_param.shape, state_dict2[key1][key].shape,key)
            #set_attr(mod, ksubmod_names, dict_param)


def set_module(mod, state_dict):
    for key1 in state_dict.keys():
        if key1 not in ['generator', 'kp_detector']:
            continue
        for key, dict_param in state_dict[key1].items():
            if key1 == 'kp_detector':
                submod_names = ['kp_extractor'] + key.split(".")
            else:
                submod_names = [key1] + key.split(".")
            #curr_param = get_attr(mod, submod_names)
            # Here you can either replace the existing one
            set_attr(mod, submod_names, dict_param)
            group_name = submod_names[:-1] + ['groups']
            og_groups = get_attr_default(mod, group_name, 1)
            if og_groups != 1:
                get_attr(mod, group_name[:-1]).groups = dict_param.shape[0]

def set_gen_module(mod, state_dict):

    for key1 in state_dict.keys():
        if key1 not in ['generator']:
            continue
        for key, dict_param in state_dict[key1].items():

            submod_names = key.split(".")
            #curr_param = get_attr(mod, submod_names)
            # Here you can either replace the existing one
            # Set the groups value for the depthwise case
            set_attr(mod, submod_names, dict_param)
            group_name = submod_names[:-1] + ['groups']
            og_groups = get_attr_default(mod, group_name, 1)
            if og_groups != 1:
                get_attr(mod, group_name[:-1]).groups = dict_param.shape[0]


def set_keypoint_module(mod, state_dict):
    for key1 in state_dict.keys():
        if key1 not in ['kp_detector']:
            continue
        for key, dict_param in state_dict[key1].items():
            submod_names = key.split(".")
            #curr_param = get_attr(mod, submod_names)
            # Here you can either replace the existing one
            set_attr(mod, submod_names, dict_param)
    

def get_input_channel_importance(weight):
    in_channels = weight.shape[1]
    importances = []
    # compute the importance for each input channel
    for i_c in range(weight.shape[1]):
        channel_weight = weight.detach()[:, i_c]
        ##################### YOUR CODE STARTS HERE #####################
        importance = torch.norm(channel_weight, 'fro')

        ##################### YOUR CODE ENDS HERE #####################
        importances.append(importance.view(1))
    return torch.cat(importances)


class Node:
    def __init__(self, index, t, i, o, value):
        self.index = index
        self.type = t
        self.i = i
        self.o = o
        self.value = value
        self.before = []
        self.after = []

        self.is_tied = False
        self.tied_after = []

        self.mirror = []

    def add_tie(self, node):
        self.tied_after.append(node)
    def mark_tied(self):
        self.is_tied = True
    def add_before(self, node):
        self.before.append(node)

    def add_mirror(self, node):
        self.mirror.append(node)

    def add_after(self, node):
        self.after.append(node)


def build_graph(all_layers, names):
    # For the sake of getting this working we are going to hardcode each layer
    graph = {}
    for index in range(len(all_layers)):
        if isinstance(all_layers[index], nn.Conv2d):

            graph[index] = Node(index, 'conv', all_layers[index].weight.shape[1],
                                all_layers[index].weight.shape[0],
                                all_layers[index])
        elif isinstance(
                all_layers[index],
                nn.modules.batchnorm._BatchNorm):
            graph[index] = Node(index, 'bn', all_layers[index].weight.shape[0],
                                all_layers[index].weight.shape[0],
                                all_layers[index])
        else:
            graph[index] = Node(index)
    gotten = set()

    def get_index(name):
        gotten.add(name)
        return names.index(name)

    def add(name1, name2):
        index1 = get_index(name1)
        index2 = get_index(name2)
        graph[index1].add_after(index2)
        graph[index2].add_before(index1)

    def add_tie(name1, name2):
        index1 = get_index(name1)
        index2 = get_index(name2)
        graph[index1].add_tie(index2)
        graph[index2].mark_tied()

    def add_mirrors(name1, name2):
        index1 = get_index(name1)
        index2 = get_index(name2)
        graph[index1].add_mirror(index2)
        graph[index2].add_mirror(index1)

    def add_names(names):
        index = 1
        while index < len(names):
            add(names[index - 1], names[index])
            index += 1

    is_efficient_net = False
    for name in names:
        if 'efficientnet' in name:
            is_efficient_net = True
            break
    if is_efficient_net:
        add_names([
            'dense_motion_network.hourglass.encoder.down_blocks.0.conv',
            'dense_motion_network.hourglass.encoder.down_blocks.0.norm',
            'dense_motion_network.hourglass.encoder.down_blocks.1.conv',
            'dense_motion_network.hourglass.encoder.down_blocks.1.norm',
            'dense_motion_network.hourglass.encoder.down_blocks.2.conv',
            'dense_motion_network.hourglass.encoder.down_blocks.2.norm',
            'dense_motion_network.hourglass.encoder.down_blocks.3.conv',
            'dense_motion_network.hourglass.encoder.down_blocks.3.norm',
            'dense_motion_network.hourglass.encoder.down_blocks.4.conv',
            'dense_motion_network.hourglass.encoder.down_blocks.4.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.0.conv',
            'dense_motion_network.hourglass.decoder.up_blocks.0.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.1.conv',
            'dense_motion_network.hourglass.decoder.up_blocks.1.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.2.conv',
            'dense_motion_network.hourglass.decoder.up_blocks.2.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.3.conv',
            'dense_motion_network.hourglass.decoder.up_blocks.3.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.4.conv',
            'dense_motion_network.hourglass.decoder.up_blocks.4.norm'
        ])

        # Add the dense motion skip connections
        add('dense_motion_network.hourglass.encoder.down_blocks.3.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.1.conv')
        add('dense_motion_network.hourglass.encoder.down_blocks.2.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.2.conv')
        add('dense_motion_network.hourglass.encoder.down_blocks.1.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.3.conv')
        add('dense_motion_network.hourglass.encoder.down_blocks.0.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.4.conv')

        # Add the dense motion outputs partly (First part)
        add('dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.mask')
        add('dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.occlusion')
        add('dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.lr_occlusion')
        add('dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.hr_background_occlusion')

        add('lr_first.conv', 'lr_first.norm')
        add_names([
            'hr_first.conv', 'hr_first.norm', 'hr_down_blocks.0.conv',
            'hr_down_blocks.0.norm'
        ])
        add_names([
            'first.conv', 'first.norm', 'down_blocks.0.conv', 'down_blocks.0.norm',
            'down_blocks.1.conv', 'down_blocks.1.norm', 'bottleneck.r0.norm1',
            'bottleneck.r0.conv1', 'bottleneck.r0.norm2', 'bottleneck.r0.conv2',
            'bottleneck.r1.norm1', 'bottleneck.r1.conv1', 'bottleneck.r1.norm2',
            'bottleneck.r1.conv2', 'bottleneck.r2.norm1', 'bottleneck.r2.conv1',
            'bottleneck.r2.norm2', 'bottleneck.r2.conv2', 'bottleneck.r3.norm1',
            'bottleneck.r3.conv1', 'bottleneck.r3.norm2', 'bottleneck.r3.conv2',
            'bottleneck.r4.norm1', 'bottleneck.r4.conv1', 'bottleneck.r4.norm2',
            'bottleneck.r4.conv2', 'bottleneck.r5.norm1', 'bottleneck.r5.conv1',
            'bottleneck.r5.norm2', 'bottleneck.r5.conv2'
        ])
        add_names(['bottleneck.r5.conv2', 'efficientnet_decoder._conv_head', 'efficientnet_decoder._bn1', 'efficientnet_decoder._blocks.1._expand_conv', 'efficientnet_decoder._blocks.1._bn0', 'efficientnet_decoder._blocks.1._depthwise_conv', 'efficientnet_decoder._blocks.1._bn1', 'efficientnet_decoder._blocks.1._se_reduce', 'efficientnet_decoder._blocks.1._se_expand', 'efficientnet_decoder._blocks.1._project_conv', 'efficientnet_decoder._blocks.1._bn2', 'efficientnet_decoder._blocks.3._expand_conv', 'efficientnet_decoder._blocks.3._bn0', 'efficientnet_decoder._blocks.3._depthwise_conv', 'efficientnet_decoder._blocks.3._bn1', 'efficientnet_decoder._blocks.3._se_reduce', 'efficientnet_decoder._blocks.3._se_expand', 'efficientnet_decoder._blocks.3._project_conv', 'efficientnet_decoder._blocks.3._bn2', 'efficientnet_decoder._blocks.5._expand_conv', 'efficientnet_decoder._blocks.5._bn0', 'efficientnet_decoder._blocks.5._depthwise_conv', 'efficientnet_decoder._blocks.5._bn1', 'efficientnet_decoder._blocks.5._se_reduce', 'efficientnet_decoder._blocks.5._se_expand', 'efficientnet_decoder._blocks.5._project_conv', 'efficientnet_decoder._blocks.5._bn2', 'efficientnet_decoder._conv_stem', 'efficientnet_decoder._bn0', 'final'])

        # Features get concatted into down block
        #add_mirror('down_blocks.1.con', 'bottleneck.r0.norm1')

        # Second up block has 32 lr features added
        add('lr_first.norm', 'up_blocks.0.conv')

        # Add 2x hr down outputs to first hr up
        add('hr_down_blocks.0.norm', 'hr_up_blocks.0.conv')
        add('hr_down_blocks.0.norm', 'hr_up_blocks.0.conv')
        add_mirrors('hr_down_blocks.0.conv', 'efficientnet_decoder._blocks.3._project_conv')
        #add_mirrors('hr_first.norm', 'efficientnet_decoder._bn1')


        add_tie('bottleneck.r0.conv1', 'bottleneck.r0.conv2')
        add_tie('bottleneck.r1.conv1', 'bottleneck.r1.conv2')
        add_tie('bottleneck.r2.conv1', 'bottleneck.r2.conv2')
        add_tie('bottleneck.r3.conv1', 'bottleneck.r3.conv2')
        add_tie('bottleneck.r4.conv1', 'bottleneck.r4.conv2')
        add_tie('bottleneck.r5.conv1', 'bottleneck.r5.conv2')
        add_tie('bottleneck.r5.conv1', 'bottleneck.r0.conv2')
        for name in names:
            if 'depthwise_conv' in name:
                add_tie(name, name)
        for i in [1,3,5]:
            add_tie('efficientnet_decoder._blocks.'+str(i)+'._se_reduce', 'efficientnet_decoder._blocks.'+str(i)+'._se_expand')

    # Build graph
    elif os.environ.get('CONV_TYPE', 'regular') == 'regular':
        add_names([
            'dense_motion_network.hourglass.encoder.down_blocks.0.conv',
            'dense_motion_network.hourglass.encoder.down_blocks.0.norm',
            'dense_motion_network.hourglass.encoder.down_blocks.1.conv',
            'dense_motion_network.hourglass.encoder.down_blocks.1.norm',
            'dense_motion_network.hourglass.encoder.down_blocks.2.conv',
            'dense_motion_network.hourglass.encoder.down_blocks.2.norm',
            'dense_motion_network.hourglass.encoder.down_blocks.3.conv',
            'dense_motion_network.hourglass.encoder.down_blocks.3.norm',
            'dense_motion_network.hourglass.encoder.down_blocks.4.conv',
            'dense_motion_network.hourglass.encoder.down_blocks.4.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.0.conv',
            'dense_motion_network.hourglass.decoder.up_blocks.0.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.1.conv',
            'dense_motion_network.hourglass.decoder.up_blocks.1.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.2.conv',
            'dense_motion_network.hourglass.decoder.up_blocks.2.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.3.conv',
            'dense_motion_network.hourglass.decoder.up_blocks.3.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.4.conv',
            'dense_motion_network.hourglass.decoder.up_blocks.4.norm'
        ])

        # Add the dense motion skip connections
        add('dense_motion_network.hourglass.encoder.down_blocks.3.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.1.conv')
        add('dense_motion_network.hourglass.encoder.down_blocks.2.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.2.conv')
        add('dense_motion_network.hourglass.encoder.down_blocks.1.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.3.conv')
        add('dense_motion_network.hourglass.encoder.down_blocks.0.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.4.conv')

        # Add the dense motion outputs partly (First part)
        add('dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.mask')
        add('dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.occlusion')
        add('dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.lr_occlusion')
        add('dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.hr_background_occlusion')

        add('lr_first.conv', 'lr_first.norm')
        add_names([
            'hr_first.conv', 'hr_first.norm', 'hr_down_blocks.0.conv',
            'hr_down_blocks.0.norm'
        ])
        add_names([
            'first.conv', 'first.norm', 'down_blocks.0.conv', 'down_blocks.0.norm',
            'down_blocks.1.conv', 'down_blocks.1.norm', 'bottleneck.r0.norm1',
            'bottleneck.r0.conv1', 'bottleneck.r0.norm2', 'bottleneck.r0.conv2',
            'bottleneck.r1.norm1', 'bottleneck.r1.conv1', 'bottleneck.r1.norm2',
            'bottleneck.r1.conv2', 'bottleneck.r2.norm1', 'bottleneck.r2.conv1',
            'bottleneck.r2.norm2', 'bottleneck.r2.conv2', 'bottleneck.r3.norm1',
            'bottleneck.r3.conv1', 'bottleneck.r3.norm2', 'bottleneck.r3.conv2',
            'bottleneck.r4.norm1', 'bottleneck.r4.conv1', 'bottleneck.r4.norm2',
            'bottleneck.r4.conv2', 'bottleneck.r5.norm1', 'bottleneck.r5.conv1',
            'bottleneck.r5.norm2', 'bottleneck.r5.conv2', 'up_blocks.0.conv',
            'up_blocks.0.norm', 'up_blocks.1.conv', 'up_blocks.1.norm',
            'hr_up_blocks.0.conv', 'hr_up_blocks.0.norm', 'final'
        ])

        # Features get concatted into down block
        add('down_blocks.1.norm', 'bottleneck.r0.norm1')

        # Second up block has 32 lr features added
        add('lr_first.norm', 'up_blocks.0.conv')

        # Add 2x hr down outputs to first hr up
        add('hr_down_blocks.0.norm', 'hr_up_blocks.0.conv')
        add('hr_down_blocks.0.norm', 'hr_up_blocks.0.conv')


        add_tie('bottleneck.r0.conv1', 'bottleneck.r0.conv2')
        add_tie('bottleneck.r1.conv1', 'bottleneck.r1.conv2')
        add_tie('bottleneck.r2.conv1', 'bottleneck.r2.conv2')
        add_tie('bottleneck.r3.conv1', 'bottleneck.r3.conv2')
        add_tie('bottleneck.r4.conv1', 'bottleneck.r4.conv2')
        add_tie('bottleneck.r5.conv1', 'bottleneck.r5.conv2')
        add_tie('bottleneck.r5.conv1', 'bottleneck.r0.conv2')
    else:
        add_names([
            'dense_motion_network.hourglass.encoder.down_blocks.0.conv.depth_conv',
            'dense_motion_network.hourglass.encoder.down_blocks.0.conv.point_conv',
            'dense_motion_network.hourglass.encoder.down_blocks.0.norm',
            'dense_motion_network.hourglass.encoder.down_blocks.1.conv.depth_conv',
            'dense_motion_network.hourglass.encoder.down_blocks.1.conv.point_conv',
            'dense_motion_network.hourglass.encoder.down_blocks.1.norm',
            'dense_motion_network.hourglass.encoder.down_blocks.2.conv.depth_conv',
            'dense_motion_network.hourglass.encoder.down_blocks.2.conv.point_conv',
            'dense_motion_network.hourglass.encoder.down_blocks.2.norm',
            'dense_motion_network.hourglass.encoder.down_blocks.3.conv.depth_conv',
            'dense_motion_network.hourglass.encoder.down_blocks.3.conv.point_conv',
            'dense_motion_network.hourglass.encoder.down_blocks.3.norm',
            'dense_motion_network.hourglass.encoder.down_blocks.4.conv.depth_conv',
            'dense_motion_network.hourglass.encoder.down_blocks.4.conv.point_conv',
            'dense_motion_network.hourglass.encoder.down_blocks.4.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.0.conv.depth_conv',
            'dense_motion_network.hourglass.decoder.up_blocks.0.conv.point_conv',
            'dense_motion_network.hourglass.decoder.up_blocks.0.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.1.conv.depth_conv',
            'dense_motion_network.hourglass.decoder.up_blocks.1.conv.point_conv',
            'dense_motion_network.hourglass.decoder.up_blocks.1.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.2.conv.depth_conv',
            'dense_motion_network.hourglass.decoder.up_blocks.2.conv.point_conv',
            'dense_motion_network.hourglass.decoder.up_blocks.2.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.3.conv.depth_conv',
            'dense_motion_network.hourglass.decoder.up_blocks.3.conv.point_conv',
            'dense_motion_network.hourglass.decoder.up_blocks.3.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.4.conv.depth_conv',
            'dense_motion_network.hourglass.decoder.up_blocks.4.conv.point_conv',
            'dense_motion_network.hourglass.decoder.up_blocks.4.norm'
        ])

        # Add the dense motion skip connections
        add('dense_motion_network.hourglass.encoder.down_blocks.3.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.1.conv.depth_conv')
        add('dense_motion_network.hourglass.encoder.down_blocks.2.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.2.conv.depth_conv')
        add('dense_motion_network.hourglass.encoder.down_blocks.1.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.3.conv.depth_conv')
        add('dense_motion_network.hourglass.encoder.down_blocks.0.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.4.conv.depth_conv')

        # Add the dense motion outputs partly (First part)
        add_names(['dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.mask.depth_conv','dense_motion_network.mask.point_conv'])
        add_names(['dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.occlusion.depth_conv','dense_motion_network.occlusion.point_conv'])
        add_names(['dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.lr_occlusion.depth_conv','dense_motion_network.lr_occlusion.point_conv'])
        add_names(['dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.hr_background_occlusion.depth_conv','dense_motion_network.hr_background_occlusion.point_conv'])

        add_names(['lr_first.conv.depth_conv', 'lr_first.conv.point_conv', 'lr_first.norm'])
        add_names([
            'hr_first.conv.depth_conv', 'hr_first.conv.point_conv',  'hr_first.norm', 'hr_down_blocks.0.conv.depth_conv','hr_down_blocks.0.conv.point_conv',
            'hr_down_blocks.0.norm'
        ])
        add_names([
            'first.conv.depth_conv', 'first.conv.point_conv', 'first.norm', 'down_blocks.0.conv.depth_conv','down_blocks.0.conv.point_conv', 'down_blocks.0.norm',
            'down_blocks.1.conv.depth_conv','down_blocks.1.conv.point_conv', 'down_blocks.1.norm', 'bottleneck.r0.norm1',
            'bottleneck.r0.conv1.depth_conv','bottleneck.r0.conv1.point_conv', 'bottleneck.r0.norm2', 'bottleneck.r0.conv2.depth_conv','bottleneck.r0.conv2.point_conv',
            'bottleneck.r1.norm1', 'bottleneck.r1.conv1.depth_conv','bottleneck.r1.conv1.point_conv', 'bottleneck.r1.norm2',
            'bottleneck.r1.conv2.depth_conv','bottleneck.r1.conv2.point_conv', 'bottleneck.r2.norm1', 'bottleneck.r2.conv1.depth_conv','bottleneck.r2.conv1.point_conv',
            'bottleneck.r2.norm2', 'bottleneck.r2.conv2.depth_conv','bottleneck.r2.conv2.point_conv', 'bottleneck.r3.norm1',
            'bottleneck.r3.conv1.depth_conv','bottleneck.r3.conv1.point_conv', 'bottleneck.r3.norm2', 'bottleneck.r3.conv2.depth_conv','bottleneck.r3.conv2.point_conv',
            'bottleneck.r4.norm1', 'bottleneck.r4.conv1.depth_conv','bottleneck.r4.conv1.point_conv', 'bottleneck.r4.norm2',
            'bottleneck.r4.conv2.depth_conv','bottleneck.r4.conv2.point_conv', 'bottleneck.r5.norm1', 'bottleneck.r5.conv1.depth_conv','bottleneck.r5.conv1.point_conv',
            'bottleneck.r5.norm2', 'bottleneck.r5.conv2.depth_conv','bottleneck.r5.conv2.point_conv', 'up_blocks.0.conv.depth_conv','up_blocks.0.conv.point_conv',
            'up_blocks.0.norm', 'up_blocks.1.conv.depth_conv','up_blocks.1.conv.point_conv', 'up_blocks.1.norm',
            'hr_up_blocks.0.conv.depth_conv','hr_up_blocks.0.conv.point_conv', 'hr_up_blocks.0.norm', 'final.depth_conv', 'final.point_conv'
        ])

        # Features get concatted into down block
        add('down_blocks.1.norm', 'bottleneck.r0.norm1')

        # Second up block has 32 lr features added
        add('lr_first.norm', 'up_blocks.0.conv.depth_conv')

        # Add 2x hr down outputs to first hr up
        add('hr_down_blocks.0.norm', 'hr_up_blocks.0.conv.depth_conv')
        add('hr_down_blocks.0.norm', 'hr_up_blocks.0.conv.depth_conv')


        add_tie('bottleneck.r0.conv1.depth_conv', 'bottleneck.r0.conv2.point_conv')
        add_tie('bottleneck.r1.conv1.depth_conv', 'bottleneck.r1.conv2.point_conv')
        add_tie('bottleneck.r2.conv1.depth_conv', 'bottleneck.r2.conv2.point_conv')
        add_tie('bottleneck.r3.conv1.depth_conv', 'bottleneck.r3.conv2.point_conv')
        add_tie('bottleneck.r4.conv1.depth_conv', 'bottleneck.r4.conv2.point_conv')
        add_tie('bottleneck.r5.conv1.depth_conv', 'bottleneck.r5.conv2.point_conv')
        #add_tie('bottleneck.r5.conv1', 'bottleneck.r0.conv2')

        # Add ties from every conv to itself's depthwise because that is what depthwise means
        # (Inputs = Outputs)

        # Take each depth conv and add it to itself
        for name in names:
            if 'depth_conv' in name:
                add_tie(name, name)


    return graph


@torch.no_grad()
def apply_channel_sorting(model):
    model = copy.deepcopy(model)  # do not modify the original model
    # fetch all the conv and bn layers from the backbone
    all_convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]

    all_convs = all_convs[:10] + all_convs[14:]
    all_bns = [
        m for m in model.modules() if isinstance(
            m, nn.modules.batchnorm._BatchNorm)
    ]
    all_layers = [
        m for m in model.modules() if (isinstance(m, nn.Conv2d) or isinstance(
            m, nn.modules.batchnorm._BatchNorm))
    ]

    layer_graph = build_graph(all_layers, [
        n for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d) or isinstance(
            m, nn.modules.batchnorm._BatchNorm))
    ])

    for base_index in layer_graph.keys():
        node = layer_graph[base_index]
        if node.type == 'conv':
            # Find the first following conv.
            if len(node.after) == 0:
                continue

            prev = node.index
            curr_node_index = node.after[0]
            curr_node = layer_graph[curr_node_index]
            found = True
            target = (0, node.o)
            while curr_node.type != 'conv':
                if len(curr_node.after) == 0:
                    found = False
                    break
                prev = curr_node.index
                curr_node_index = curr_node.after[0]
                curr_node = layer_graph[curr_node_index]
                counter = 0

                for prev_node in curr_node.before:
                    if prev_node == prev:
                        target = (counter, counter + (target[1] - target[0]))
                        break
                    counter += layer_graph[prev_node].o

            if not found:
                continue

            # Now curr_node points to the first conv input node for this
            # Find the actual elements you care about
            counter = 0

            # Possible flip here
            important_elements = curr_node.value.weight[:, target[0]:target[1]]
            importance = get_input_channel_importance(important_elements)
            # sorting from large to small
            sort_idx = torch.argsort(importance, descending=True)

            # Sort the outputs of the actual node
            prev_conv = node
            prev_conv.value.weight.copy_(
                torch.index_select(prev_conv.value.weight.detach(), 0,
                                   sort_idx))
            prev_conv.value.bias.copy_(
                torch.index_select(prev_conv.value.bias.detach(), 0, sort_idx))

            # layer_graph[temp.index].value.weight.copy_(torch.index_select(
            #     layer_graph[temp.index].value.weight.detach(), 0, sort_idx))
            # layer_graph[temp.index].value.bias.copy_(torch.index_select(
            #     layer_graph[temp.index].bias.detach(), 0, sort_idx))

            def follow(node, layer_graph, previous, p_indices):
                # Find the elements corresponding to this node

                indices = []
                counter = 0
                for b in node.before:
                    if b == previous:
                        for p_index in p_indices:
                            indices.append(
                                (counter + p_index[0], counter + p_index[1]))

                    counter += layer_graph[b].o

                def shrink_bn(bn, tensor_name):
                    thing_you_are_changing = []
                    for index in indices:
                        nvd = getattr(node.value, tensor_name).detach()
                        nvd = nvd[index[0]:index[1]]
                        nvd.set_(torch.index_select(nvd, 0, sort_idx))
                        thing_you_are_changing.append(nvd)

                    if len(thing_you_are_changing) == 0:
                        print("Somehow we didnt change anything")

                    starter = [
                        getattr(node.value,
                                tensor_name).detach()[:indices[0][0]]
                    ]
                    for index in range(len(indices)):
                        starter.append(thing_you_are_changing[index])
                        if len(thing_you_are_changing) > index + 1:
                            starter.append(
                                getattr(node.value, tensor_name).detach()
                                [indices[index][1]:indices[index + 1][0]])
                        else:
                            starter.append(
                                getattr(
                                    node.value,
                                    tensor_name).detach()[indices[index][1]:])

                    tensor_to_apply = getattr(bn.value, tensor_name)
                    tensor_to_apply.copy_(torch.cat(starter).clone().detach())
                    #node.value.weight.set_(torch.cat(starter).clone().detach())

                if node.type == 'conv':
                    thing_you_are_changing = []
                    for index in indices:
                        nvd = node.value.weight.detach()
                        nvd = nvd[:, index[0]:index[1]]
                        nvd.set_(torch.index_select(nvd, 1, sort_idx))
                        thing_you_are_changing.append(nvd)

                    if len(thing_you_are_changing) == 0:
                        print("Somehow we didnt change anything")

                    starter = [node.value.weight.detach()[:, :indices[0][0]]]
                    for index in range(len(indices)):
                        starter.append(thing_you_are_changing[index])
                        if len(thing_you_are_changing) > index + 1:
                            starter.append(
                                node.value.weight.detach()
                                [:, indices[index][1]:indices[index + 1][0]])
                        else:
                            starter.append(
                                node.value.weight.detach()[:,
                                                           indices[index][1]:])

                    tensor_to_apply = getattr(node.value, 'weight')
                    tensor_to_apply.copy_(
                        torch.cat(starter, dim=1).clone().detach())
                    #layer_graph[node.index].value.weight.set_(tensor_to_apply)

                    return

                else:

                    for tensor_name in [
                            'weight', 'bias', 'running_mean', 'running_var'
                    ]:
                        shrink_bn(node, tensor_name)

                for next_layer in set(node.after):
                    follow(layer_graph[next_layer], layer_graph, node.index,
                           indices)

            for node_after in set(node.after):
                follow(layer_graph[node_after], layer_graph, node.index,
                       [(0, node.o)])

    return model


def get_generator_time(model, x):
    #for i in range(10):
    #    _ = model(inp)
    driving_lr =  x.get('driving_lr', None)

    kp_source = model.kp_extractor(x['source'])
    
            
    if driving_lr is not None:
        kp_driving = model.kp_extractor(driving_lr)
    else:
        kp_driving = model.kp_extractor(x['driving'])
    

    #warmup
    #model = torch.compile(model)
    for _ in range(10):
        generated = model.generator(x['source'], kp_source=kp_source, 
                kp_driving=kp_driving, update_source=True, 
                driving_lr=driving_lr)
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_time = 0
    for i in range(50):
        starter.record()
        with torch.no_grad():
            generated = model.generator(x['source'], kp_source=kp_source, 
                    kp_driving=kp_driving, update_source=True, 
                    driving_lr=driving_lr)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        total_time += curr_time

    return total_time/50

def get_gen_input(model=None, x=None):
    if not x is None:
        driving_lr =  x.get('driving_lr', None)

        kp_source = model.kp_extractor(x['source'])
        
                
        if driving_lr is not None:
            kp_driving = model.kp_extractor(driving_lr)
        else:
            kp_driving = model.kp_extractor(x['driving'])
        
        get_gen_input.inputs = (x['source'], kp_source, kp_driving, True, driving_lr)
    return get_gen_input.inputs

def calculate_macs(model, file_name = None):

    inputs = get_gen_input()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = profile_macs(model, inputs, reduction=None)
    return result

def total_macs(model):

    macs_dict = calculate_macs(model)
    return sum(macs_dict.values())



@torch.no_grad()
def channel_prune(model, deletions):
    """Apply channel pruning to each of the conv layer in the backbone
    Note that for prune_ratio, we can either provide a floating-point number,
    indicating that we use a uniform pruning rate for all layers, or a list of
    numbers to indicate per-layer pruning rate.
    """
    # sanity check of provided prune_ratio
    model = copy.deepcopy(model)

    all_layers = [
        m for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d) or isinstance(
            m, nn.modules.batchnorm._BatchNorm))
    ]
    layer_graph = build_graph(all_layers, [
        n for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d) or isinstance(
            m, nn.modules.batchnorm._BatchNorm))
    ])

    for index, pruners in deletions.items():
        node = layer_graph[index]
        if pruners[0] == 'first':
            if node.type == 'conv' and len(node.after) != 0:
                # Prune the outupts
                node.value.weight.set_(
                    node.value.weight.detach()[:pruners[1][0]].contiguous())
                if node.value.bias is not None:
                    node.value.bias.set_(
                        node.value.bias.detach()[:pruners[1][0]].contiguous())
            else:
                print("Should not be shrinking a batchnorm")
            if node.value.groups != 1:
                node.value.groups = node.value.weight.shape[0]
        else:
            for i in range(len(pruners)):
                prune_indices = pruners[len(pruners) - i - 1]

                if node.type == 'conv':
                    nvd = node.value.weight.detach()
                    nvd = torch.cat([
                        nvd[:, :prune_indices[0]], nvd[:, prune_indices[1]:]
                    ],
                                    dim=1)
                    node.value.weight.set_(nvd.clone().detach().contiguous())
                if node.type == 'bn':

                    def f_set(nvd, w, prune_indices):
                        nvd = torch.cat([
                            nvd[:prune_indices[0]], nvd[prune_indices[1]:]
                        ])
                        w.set_(nvd.clone().detach())

                    f_set(node.value.weight.detach(), node.value.weight,
                          prune_indices)
                    f_set(node.value.bias.detach(), node.value.bias,
                          prune_indices)
                    f_set(node.value.running_mean.detach(),
                          node.value.running_mean, prune_indices)
                    f_set(node.value.running_var.detach(),
                          node.value.running_var, prune_indices)

    return model


def get_metrics_loss(metrics_dataloader, lr_size, generator_full, generator_type):
    total_loss = 0
    with torch.no_grad():
        for y in metrics_dataloader:
            y['driving_lr'] = F.interpolate(y['driving'], lr_size)
            for k in y:
                try:
                    y[k] = y[k].cuda()
                except:
                    pass
            losses_generator, metrics_generated = generator_full(
                y, generator_type)
            loss_values = [val.mean() for val in losses_generator.values()]
            loss = sum(loss_values)
            total_loss += loss.item()

    return total_loss


def compute_deletion(layer_graph, custom_deletions, deleted_things, layer, custom=None, reason=None):
    """
    Given a layer, generate the list of the indexes we need to delete from its following layers
    """
    # Reduct its output
    curr_output = layer_graph[layer].o

    # find all the following layers
    following_layers = []

    def follow(x):
        following_layers.append(x.index)
        if x.type == 'conv':
            return
        for y in x.after:
            follow(layer_graph[y])

    for after_layer in layer_graph[layer].after:
        follow(layer_graph[after_layer])



    # If you delete a part of the previous one, not just all of it then you cannot just say these n are to be deleted
    # So for each layer you need to figure out what you are deleting from it, then for the next layer figure out what is deleted

    # So make a map of deletions
    deletions = {}
    if custom is None:
        deletions[layer] = [(layer_graph[layer].o - i, layer_graph[layer].o)]
    else:
        deletions[layer] = [(layer_graph[layer].o - custom, layer_graph[layer].o)]

    amount_to_delete = custom
    if custom is None:
        amount_to_delete = i

    
    for mirror in layer_graph[layer].mirror:
        # Start a new custom deletion for the mirror
        if reason != 'mirror':
            amount_to_delete = amount_to_delete
            custom_deletions.append((mirror, amount_to_delete, 'mirror'))
            print("Starting a mirrorred deletion")


    for following_layer in following_layers:
        if following_layer in deletions:
            continue

        deletions[following_layer] = []
        counter = 0
        #previous_mirrors = [i for i in layer_graph[following_layer].before
        for previous_layer in layer_graph[following_layer].before:
            if previous_layer not in deletions:
                continue
            # Currently assumes we can only have one set of mirrors for a particular mirror output
            # If the previous layer is a mirror with the next one, exclude it so we only include it once
            if len(layer_graph[previous_layer].mirror) == 0 or True or min(layer_graph[previous_layer].mirror) > previous_layer:
                for deletion in deletions[previous_layer]:
                    deletions[following_layer].append(
                        (counter + deletion[0], counter + deletion[1]))
                counter += layer_graph[previous_layer].o
            else:
                print("ignoring layer", previous_layer, "in", layer_graph[previous_layer].mirror)

        if len(layer_graph[following_layer].tied_after) != 0:
            total_deleted = 0
            for del_tuple in deletions[following_layer]:
                total_deleted += del_tuple[1] - del_tuple[0]
            for after_tied in layer_graph[following_layer].tied_after:
                if after_tied not in deleted_things:
                    deleted_things.add(after_tied)
                    custom_deletions.append((after_tied, total_deleted, 'tie'))
            


    # The original layer is a special case because you delete its output not input
    deletions[layer].insert(0, 'first')
    return deletions


def try_reduce(curr_loss, curr_model, per_layer_macs, dataloader, layer_graph, layer, kp_detector, discriminator, train_params, model, target, current, lr_size, generator_type, metrics_dataloader, generator_full):
    custom_deletions = []
    deleted_things = set()
    model_copy = copy.deepcopy(model)
    if 1 >= layer_graph[layer].o:
        return None, None
    deletions = compute_deletion(layer_graph, custom_deletions, deleted_things,layer, 1)
    print(deletions)
    if deletions is None:
        return None, None
    model_copy = channel_prune( model_copy, deletions)
    while len(custom_deletions) != 0:
        custom_deletion =  custom_deletions[0]
        custom_deletions = custom_deletions[1:]
        deletions = compute_deletion(layer_graph, custom_deletions, deleted_things, custom_deletion[0], custom_deletion[1], custom_deletion[2])
        print(deletions)
        model_copy = channel_prune( model_copy, deletions)

    after_1_reduce = total_macs(model_copy)
    print(after_1_reduce)
    if after_1_reduce == current:
        print("Trying to remove something that is not a part of the model")
        return None, None

    to_remove = int((current-target) // (current - after_1_reduce))

    # Check the validity of a deletion op
    if to_remove >= layer_graph[layer].o:
        print("Cannot remove enough to hit target")
        return None, None
    
    if to_remove >= 1:
        model_copy = copy.deepcopy(model)
        deleted_things.clear()
        custom_deletions = []
        deletions = compute_deletion(layer_graph, custom_deletions, deleted_things, layer, to_remove)
        if deletions is None:
            return None, None
        model_copy = channel_prune( model_copy, deletions)
        print(deletions)
        while len(custom_deletions) != 0:
            custom_deletion =  custom_deletions[0]
            custom_deletions = custom_deletions[1:]
            deletions = compute_deletion(layer_graph, custom_deletions, deleted_things,custom_deletion[0], custom_deletion[1], custom_deletion[2])
            print(deletions)
            model_copy = channel_prune( model_copy, deletions)


    print("done")

    # Train
    old_model = generator_full.generator
    generator_full.generator = model_copy
    optimizer_generator = torch.optim.Adam(generator_full.generator.parameters(),
                                           lr=train_params['lr_generator'],
                                           betas=(0.5, 0.999))
    #if torch.cuda.is_available():
    #    generator_full = DataParallelWithCallback(generator_full,
    #                                              device_ids=[0])

    counter = 0
    for k in dataloader:
        break


    #generator_full.generator = old_model
    c=0
    for x in tqdm(dataloader):
        c += 1
        x['driving_lr'] = F.interpolate(x['driving'], lr_size)
        for k in x:
            try:
                x[k] = x[k].cuda()
            except:
                pass
        losses_generator, generated = generator_full(x, generator_type)
        loss_values = [val.mean() for val in losses_generator.values()]
        loss = sum(loss_values)
        loss.backward()
        optimizer_generator.step()
        optimizer_generator.zero_grad()
    total_loss= get_metrics_loss(metrics_dataloader, lr_size, generator_full, generator_type)
    print("Loss for this model is: ", total_loss)
    

    generator_full.generator = old_model


    #del model_copy
    #del optimizer_generator
    # Store the best module
    if curr_loss is None or total_loss < curr_loss:
        return total_loss, model_copy
    else:
        return None, None
    torch.cuda.empty_cache()

def reduce_macs(model, target, current, kp_detector, discriminator,
                                        train_params, dataloader, metrics_dataloader, generator_type, lr_size, generator_full):
    # Take each layer and reduce its macs
    all_layers = [
        m for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d) or isinstance(
            m, nn.modules.batchnorm._BatchNorm))
    ]
    layer_graph = build_graph(all_layers, [
        n for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d) or isinstance(
            m, nn.modules.batchnorm._BatchNorm))
    ])
    per_layer_macs = calculate_macs(model)
    deletions = []
    curr_model = None
    curr_loss = None
    i = 0
    for layer in layer_graph:
        i += 1
        if layer_graph[layer].type != 'conv':
            continue
        if layer_graph[layer].is_tied:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss, t_model = try_reduce(curr_loss, curr_model, per_layer_macs, dataloader, layer_graph, layer, kp_detector, discriminator, train_params, model, target, current, lr_size, generator_type, metrics_dataloader, generator_full)
        #torch.cuda.empty_cache()
        if loss is not None:
            print("Updated model")
            curr_model = t_model
            curr_loss = loss

    if curr_model is None:
        print("Could not shrink anymore")
        raise StopIteration("Modle shrinking complete")
        return model
    return curr_model
