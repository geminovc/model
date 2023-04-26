from tqdm import trange
import gc
from utils import get_decode_and_bottleneck_macs
import os
from tqdm import tqdm
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import torch
import torch.nn.functional as F
import first_order_model
from torchprofile import profile_macs

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

def move_to_gpu(data):
    """
    Move every value in data to gpu if possible.
    Operates in place.
    """
    for key in data:
        try:
            data[key] = data[key].cuda()
        except:
            pass


def get_attr(obj, names):
    """
    Get the value of obj.names[0].names[1]...
    """
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


def get_attr_default(obj, names, default):
    """
    Get the value of obj.names[0].names[1]...

    If final parameter doesn't exist, use default
    """
    if len(names) == 1:
        return getattr(obj, names[0], default)
    else:
        return get_attr_default(getattr(obj, names[0], default), names[1:],
                                default)


def set_attr(obj, names, val):
    """
    Set the value of obj.names[0].names[1]... to val
    """
    if len(names) == 0:
        with torch.no_grad():
            obj.set_(val)
        #setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def print_gen_module(state_dict):
    """
    Given a state dict for the full generator, print the gen module
    Primarily used when initially coding up netadapt
    """
    for outer in state_dict.keys():
        if outer not in ['generator']:
            continue
        for key, dict_param in state_dict[outer].items():
            if outer == 'kp_detector':
                submod_names = ['kp_extractor'] + key.split(".")
            else:
                submod_names = [outer] + key.split(".")

            if 'norm' not in key and 'bias' not in key:
                print(dict_param.shape, key)


def print_diff(state_dict, state_dict2):
    """
    Prints the git-diff style diff of the different parameters shapes in the generator module of state_dicts.

    Used when manually comparing two netadapted models
    """

    for outer in state_dict.keys():
        if outer not in ['generator']:
            continue
        for key, dict_param in state_dict[outer].items():
            if 'norm' not in key and 'bias' not in key:
                if state_dict2[outer][key].shape != dict_param.shape:

                    print(dict_param.shape, state_dict2[outer][key].shape, key)


def set_module(mod, state_dict, force_model=None):
    """
    Given a generator full model, set the generator and keypoint detector with state dict.
    This is different from set state dict since it goes in and edits weights regardless of shape mismatch. Used only in loading netadapted parameters
    """

    for outer in state_dict.keys():
        if outer not in ['generator', 'kp_detector', 'discriminator']:
            continue
        if force_model is not None and outer != force_model:
            continue
        for key, dict_param in state_dict[outer].items():
            if outer == 'kp_detector':
                submod_names = ['kp_extractor'] + key.split(".")
            else:
                submod_names = [outer] + key.split(".")
            if force_model != None:
                submod_names = key.split(".")
            # Here you can either replace the existing one
            set_attr(mod, submod_names, dict_param)
            group_name = submod_names[:-1] + ['groups']
            og_groups = get_attr_default(mod, group_name, 1)
            if og_groups != 1:
                get_attr(mod, group_name[:-1]).groups = dict_param.shape[0]
                try:
                    get_attr(mod, submod_names[:-1]).in_channels = dict_param.shape[0]
                    get_attr(mod, submod_names[:-1]).out_channels = dict_param.shape[0]
                except:
                    pass
            else:
                try:
                    get_attr(mod, submod_names[:-1]).in_channels = dict_param.shape[1]
                    get_attr(mod, submod_names[:-1]).out_channels = dict_param.shape[0]
                except:
                    pass



def set_gen_module(mod, state_dict):
    """
    See: set_module
    applies only to generators
    """
    set_module(mod, state_dict, 'generator')

def set_keypoint_module(mod, state_dict):
    """
    See: set module
    applies only to keypoint detectors
    """
    set_module(mod, state_dict, 'kp_detector')

class Node:
    """
    Stores the information each node in the computation graph for a model will need.
    Used in dependency graph of netadapt.

    Tied nodes are used when the inputs of a layer are tied to the outputs of a following layer.
    This happens in the resnet when the inputs of a block and the outputs of a block must
    have the same shape and when deleting from one layer, you should also delete from the second.

    Mirrored nodes are nodes whose outputs are directly added together, so when deleting from one, you should
    also delete from the other. I added this only when I needed it in distillation, so I could convert all the tied nodes to mirror, but its painful to go through that process agian.
    """
    def __init__(self, index, t, i, o, value, name):
        self.name = name
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
    """
    Manually build dependency graph
    """

    # Create all the nodes
    graph = {}
    for index in range(len(all_layers)):
        if isinstance(all_layers[index], nn.Conv2d):

            graph[index] = Node(index, 'conv',
                                all_layers[index].weight.shape[1],
                                all_layers[index].weight.shape[0],
                                all_layers[index], names[index])
        elif isinstance(all_layers[index], nn.modules.batchnorm._BatchNorm):
            graph[index] = Node(index, 'bn', all_layers[index].weight.shape[0],
                                all_layers[index].weight.shape[0],
                                all_layers[index], names[index])
        else:
            graph[index] = Node(index)
    gotten = set()

    def get_index(name):
        gotten.add(name)
        return names.index(name)

    def add(name1, name2):
        """
        Adds a connection where name1.output goes into name2.input
        """
        index1 = get_index(name1)
        index2 = get_index(name2)
        graph[index1].add_after(index2)
        graph[index2].add_before(index1)

    def add_tie(name1, name2):
        """
        Add a tie where name1.input = name2.output shape
        Used in resnet
        """
        index1 = get_index(name1)
        index2 = get_index(name2)
        graph[index1].add_tie(index2)
        graph[index2].mark_tied()

    def add_mirrors(name1, name2):
        """
        Add a mirror where name1.output shape = name2.output shape
        Used when concat is replaced with add (in distilled network)
        """
        index1 = get_index(name1)
        index2 = get_index(name2)
        graph[index1].add_mirror(index2)
        graph[index2].add_mirror(index1)

    def add_names(names):
        """
        Extended version of add which takes in a list
        """
        index = 1
        while index < len(names):
            add(names[index - 1], names[index])
            index += 1

    is_efficient_net = False
    for name in names:
        if 'efficientnet' in name:
            is_efficient_net = True
            break

    is_1024 = False
    for name in names:
        if 'hr_down_blocks.1' in name:
            is_1024 = True
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
            'first.conv', 'first.norm', 'down_blocks.0.conv',
            'down_blocks.0.norm', 'down_blocks.1.conv', 'down_blocks.1.norm',
            'bottleneck.r0.norm1', 'bottleneck.r0.conv1',
            'bottleneck.r0.norm2', 'bottleneck.r0.conv2',
            'bottleneck.r1.norm1', 'bottleneck.r1.conv1',
            'bottleneck.r1.norm2', 'bottleneck.r1.conv2',
            'bottleneck.r2.norm1', 'bottleneck.r2.conv1',
            'bottleneck.r2.norm2', 'bottleneck.r2.conv2',
            'bottleneck.r3.norm1', 'bottleneck.r3.conv1',
            'bottleneck.r3.norm2', 'bottleneck.r3.conv2',
            'bottleneck.r4.norm1', 'bottleneck.r4.conv1',
            'bottleneck.r4.norm2', 'bottleneck.r4.conv2',
            'bottleneck.r5.norm1', 'bottleneck.r5.conv1',
            'bottleneck.r5.norm2', 'bottleneck.r5.conv2'
        ])
        add_names([
            'bottleneck.r5.conv2', 'efficientnet_decoder._conv_head',
            'efficientnet_decoder._bn1',
            'efficientnet_decoder._blocks.1._expand_conv',
            'efficientnet_decoder._blocks.1._bn0',
            'efficientnet_decoder._blocks.1._depthwise_conv',
            'efficientnet_decoder._blocks.1._bn1',
            'efficientnet_decoder._blocks.1._se_reduce',
            'efficientnet_decoder._blocks.1._se_expand',
            'efficientnet_decoder._blocks.1._project_conv',
            'efficientnet_decoder._blocks.1._bn2',
            'efficientnet_decoder._blocks.3._expand_conv',
            'efficientnet_decoder._blocks.3._bn0',
            'efficientnet_decoder._blocks.3._depthwise_conv',
            'efficientnet_decoder._blocks.3._bn1',
            'efficientnet_decoder._blocks.3._se_reduce',
            'efficientnet_decoder._blocks.3._se_expand',
            'efficientnet_decoder._blocks.3._project_conv',
            'efficientnet_decoder._blocks.3._bn2',
            'efficientnet_decoder._blocks.5._expand_conv',
            'efficientnet_decoder._blocks.5._bn0',
            'efficientnet_decoder._blocks.5._depthwise_conv',
            'efficientnet_decoder._blocks.5._bn1',
            'efficientnet_decoder._blocks.5._se_reduce',
            'efficientnet_decoder._blocks.5._se_expand',
            'efficientnet_decoder._blocks.5._project_conv',
            'efficientnet_decoder._blocks.5._bn2',
            'efficientnet_decoder._conv_stem', 'efficientnet_decoder._bn0',
            'final'
        ])

        # Features get concatted into down block

        # Second up block has 32 lr features added
        add('lr_first.norm', 'up_blocks.0.conv')

        # Add 2x hr down outputs to first hr up
        add('hr_down_blocks.0.norm', 'hr_up_blocks.0.conv')
        add('hr_down_blocks.0.norm', 'hr_up_blocks.0.conv')
        add_mirrors('hr_down_blocks.0.conv',
                    'efficientnet_decoder._blocks.3._project_conv')

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
        for i in [1, 3, 5]:
            add_tie('efficientnet_decoder._blocks.' + str(i) + '._se_reduce',
                    'efficientnet_decoder._blocks.' + str(i) + '._se_expand')

    # Build graph for default model
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

        if is_1024:
            add_names(['hr_down_blocks.0.norm', 'hr_down_blocks.1.conv', 'hr_down_blocks.1.norm'])

        add_names([
            'first.conv', 'first.norm', 'down_blocks.0.conv',
            'down_blocks.0.norm', 'down_blocks.1.conv', 'down_blocks.1.norm',
            'bottleneck.r0.norm1', 'bottleneck.r0.conv1',
            'bottleneck.r0.norm2', 'bottleneck.r0.conv2',
            'bottleneck.r1.norm1', 'bottleneck.r1.conv1',
            'bottleneck.r1.norm2', 'bottleneck.r1.conv2',
            'bottleneck.r2.norm1', 'bottleneck.r2.conv1',
            'bottleneck.r2.norm2', 'bottleneck.r2.conv2',
            'bottleneck.r3.norm1', 'bottleneck.r3.conv1',
            'bottleneck.r3.norm2', 'bottleneck.r3.conv2',
            'bottleneck.r4.norm1', 'bottleneck.r4.conv1',
            'bottleneck.r4.norm2', 'bottleneck.r4.conv2',
            'bottleneck.r5.norm1', 'bottleneck.r5.conv1',
            'bottleneck.r5.norm2', 'bottleneck.r5.conv2', 'up_blocks.0.conv',
            'up_blocks.0.norm', 'up_blocks.1.conv', 'up_blocks.1.norm',
            'hr_up_blocks.0.conv', 'hr_up_blocks.0.norm'
        ])
        if is_1024:
            add_names(['hr_up_blocks.0.norm', 'hr_up_blocks.1.conv', 'hr_up_blocks.1.norm', 'final'])
        else:
            add('hr_up_blocks.0.norm', 'final')

        # Features get concatted into down block
        add('down_blocks.1.norm', 'bottleneck.r0.norm1')


        if is_1024:
            # Second up block has 32 lr features added
            add('lr_first.norm', 'up_blocks.1.conv')
        else:
            # Second up block has 32 lr features added
            add('lr_first.norm', 'up_blocks.0.conv')

        # Add 2x hr down outputs to first hr up

        if is_1024:
            add('hr_down_blocks.1.norm', 'hr_up_blocks.0.conv')
            add('hr_down_blocks.1.norm', 'hr_up_blocks.0.conv')
            add('hr_down_blocks.0.norm', 'hr_up_blocks.1.conv')
            add('hr_down_blocks.0.norm', 'hr_up_blocks.1.conv')
        else:
            add('hr_down_blocks.0.norm', 'hr_up_blocks.0.conv')
            add('hr_down_blocks.0.norm', 'hr_up_blocks.0.conv')

        add_tie('bottleneck.r0.conv1', 'bottleneck.r0.conv2')
        add_tie('bottleneck.r1.conv1', 'bottleneck.r1.conv2')
        add_tie('bottleneck.r2.conv1', 'bottleneck.r2.conv2')
        add_tie('bottleneck.r3.conv1', 'bottleneck.r3.conv2')
        add_tie('bottleneck.r4.conv1', 'bottleneck.r4.conv2')
        add_tie('bottleneck.r5.conv1', 'bottleneck.r5.conv2')
        add_tie('bottleneck.r5.conv1', 'bottleneck.r0.conv2')

    # Build graph for depthwise convolution model
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
        add(
            'dense_motion_network.hourglass.encoder.down_blocks.3.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.1.conv.depth_conv'
        )
        add(
            'dense_motion_network.hourglass.encoder.down_blocks.2.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.2.conv.depth_conv'
        )
        add(
            'dense_motion_network.hourglass.encoder.down_blocks.1.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.3.conv.depth_conv'
        )
        add(
            'dense_motion_network.hourglass.encoder.down_blocks.0.norm',
            'dense_motion_network.hourglass.decoder.up_blocks.4.conv.depth_conv'
        )

        # Add the dense motion outputs partly (First part)
        add_names([
            'dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.mask.depth_conv',
            'dense_motion_network.mask.point_conv'
        ])
        add_names([
            'dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.occlusion.depth_conv',
            'dense_motion_network.occlusion.point_conv'
        ])
        add_names([
            'dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.lr_occlusion.depth_conv',
            'dense_motion_network.lr_occlusion.point_conv'
        ])
        add_names([
            'dense_motion_network.hourglass.decoder.up_blocks.4.norm',
            'dense_motion_network.hr_background_occlusion.depth_conv',
            'dense_motion_network.hr_background_occlusion.point_conv'
        ])

        add_names([
            'lr_first.conv.depth_conv', 'lr_first.conv.point_conv',
            'lr_first.norm'
        ])
        add_names([
            'hr_first.conv.depth_conv', 'hr_first.conv.point_conv',
            'hr_first.norm', 'hr_down_blocks.0.conv.depth_conv',
            'hr_down_blocks.0.conv.point_conv', 'hr_down_blocks.0.norm'
        ])
        if is_1024:
            add_names(['hr_down_blocks.0.norm', 'hr_down_blocks.1.conv.depth_conv', 'hr_down_blocks.1.conv.point_conv', 'hr_down_blocks.1.norm'])

        add_names([
            'first.conv.depth_conv', 'first.conv.point_conv', 'first.norm',
            'down_blocks.0.conv.depth_conv', 'down_blocks.0.conv.point_conv',
            'down_blocks.0.norm', 'down_blocks.1.conv.depth_conv',
            'down_blocks.1.conv.point_conv', 'down_blocks.1.norm',
            'bottleneck.r0.norm1', 'bottleneck.r0.conv1.depth_conv',
            'bottleneck.r0.conv1.point_conv', 'bottleneck.r0.norm2',
            'bottleneck.r0.conv2.depth_conv', 'bottleneck.r0.conv2.point_conv',
            'bottleneck.r1.norm1', 'bottleneck.r1.conv1.depth_conv',
            'bottleneck.r1.conv1.point_conv', 'bottleneck.r1.norm2',
            'bottleneck.r1.conv2.depth_conv', 'bottleneck.r1.conv2.point_conv',
            'bottleneck.r2.norm1', 'bottleneck.r2.conv1.depth_conv',
            'bottleneck.r2.conv1.point_conv', 'bottleneck.r2.norm2',
            'bottleneck.r2.conv2.depth_conv', 'bottleneck.r2.conv2.point_conv',
            'bottleneck.r3.norm1', 'bottleneck.r3.conv1.depth_conv',
            'bottleneck.r3.conv1.point_conv', 'bottleneck.r3.norm2',
            'bottleneck.r3.conv2.depth_conv', 'bottleneck.r3.conv2.point_conv',
            'bottleneck.r4.norm1', 'bottleneck.r4.conv1.depth_conv',
            'bottleneck.r4.conv1.point_conv', 'bottleneck.r4.norm2',
            'bottleneck.r4.conv2.depth_conv', 'bottleneck.r4.conv2.point_conv',
            'bottleneck.r5.norm1', 'bottleneck.r5.conv1.depth_conv',
            'bottleneck.r5.conv1.point_conv', 'bottleneck.r5.norm2',
            'bottleneck.r5.conv2.depth_conv', 'bottleneck.r5.conv2.point_conv',
            'up_blocks.0.conv.depth_conv', 'up_blocks.0.conv.point_conv',
            'up_blocks.0.norm', 'up_blocks.1.conv.depth_conv',
            'up_blocks.1.conv.point_conv', 'up_blocks.1.norm',
            'hr_up_blocks.0.conv.depth_conv', 'hr_up_blocks.0.conv.point_conv',
            'hr_up_blocks.0.norm'])

        if is_1024:
            add_names(['hr_up_blocks.0.norm', 'hr_up_blocks.1.conv.depth_conv', 'hr_up_blocks.1.conv.point_conv', 'hr_up_blocks.1.norm', 'final.depth_conv', 'final.point_conv'])
        else:
            add_names(['hr_up_blocks.0.norm', 'final.depth_conv', 'final.point_conv'])

        # Features get concatted into down block
        add('down_blocks.1.norm', 'bottleneck.r0.norm1')

        if is_1024:
            # Second up block has 32 lr features added
            add('lr_first.norm', 'up_blocks.1.conv.depth_conv')
        else:
            # Second up block has 32 lr features added
            add('lr_first.norm', 'up_blocks.0.conv.depth_conv')

        if is_1024:
            # Add 2x hr down outputs to first hr up
            add('hr_down_blocks.1.norm', 'hr_up_blocks.0.conv.depth_conv')
            add('hr_down_blocks.1.norm', 'hr_up_blocks.0.conv.depth_conv')
            add('hr_down_blocks.0.norm', 'hr_up_blocks.1.conv.depth_conv')
            add('hr_down_blocks.0.norm', 'hr_up_blocks.1.conv.depth_conv')
        else:
            # Add 2x hr down outputs to first hr up
            add('hr_down_blocks.0.norm', 'hr_up_blocks.0.conv.depth_conv')
            add('hr_down_blocks.0.norm', 'hr_up_blocks.0.conv.depth_conv')

        add_tie('bottleneck.r0.conv1.depth_conv',
                'bottleneck.r0.conv2.point_conv')
        add_tie('bottleneck.r1.conv1.depth_conv',
                'bottleneck.r1.conv2.point_conv')
        add_tie('bottleneck.r2.conv1.depth_conv',
                'bottleneck.r2.conv2.point_conv')
        add_tie('bottleneck.r3.conv1.depth_conv',
                'bottleneck.r3.conv2.point_conv')
        add_tie('bottleneck.r4.conv1.depth_conv',
                'bottleneck.r4.conv2.point_conv')
        add_tie('bottleneck.r5.conv1.depth_conv',
                'bottleneck.r5.conv2.point_conv')

        # Add ties from every conv to itself's depthwise because that is what depthwise means
        # (Inputs = Outputs)

        # Take each depth conv and add it to itself
        for name in names:
            if 'depth_conv' in name:
                add_tie(name, name)

    return graph

def pick_channels_with_lowest_importances(n, importances):
    """
    Select n channges with lowest importances. Return their indices.
    Input: n - number of channels to select
              importances - list of importances of each channel (float)
    """

    # Create a list of tuples (channel index, importance)
    channel_importances = [(i, importances[i]) for i in range(len(importances))]

    # Sort the list by importance, which is the second element in the tuple
    channel_importances = sorted(channel_importances, key=lambda x: x[1])

    # Return the indices of the n channels with lowest importances
    return [channel_importances[i][0] for i in range(n)]


def convert_to_deletions_list(indices):
    """ 
    Convert list of indices to deletion list.

    Example:
    indices = [0, 1, 3, 5]
    deletion_list = [(0, 1), (1,2), (3,4), (5,6)]

    This is used when we get the indices of the unimportant columns, then convert into a deletion list.
    """

    deletion_list = []
    i = 0
    indices = sorted(indices)
    while i < len(indices):
        old_i = i
        while i < len(indices) - 1 and indices[i] + 1 == indices[i+1]:
            i += 1
        deletion_list.append((indices[old_i], indices[i]+1))
        i += 1
    return deletion_list


def get_importances(weight, relevent_slice):
    """
    Use the L2 norm of the weights to get the importance of each channel in relevent_slice.
    """

    # Get the weights of the relevent slice
    relevent_weights = weight[:,relevent_slice[0]:relevent_slice[1]]

    # Get the norm of each channel
    importances = torch.linalg.vector_norm(relevent_weights, dim=(0,2,3))

    return importances

def get_relevent_slice(layer_graph, node, target):
    """
    Given a node and a target, finds the slices of node which are affected by target.
    Returns [(x, y), (z, n)...] where each pair is the start and end of a slice
    """
    counter = 0
    output = []
    for prev_node in layer_graph[node].before:
        if prev_node == target:
            output += [(counter, counter+layer_graph[prev_node].o)]

        counter += layer_graph[prev_node].o

    if len(output) == 0:
        counter = 0
        for prev_node in layer_graph[node].before:
            temp_output = get_relevent_slice(layer_graph, prev_node, target)
            for out in temp_output:
                output += [(out[0] + counter, out[1] + counter)]

            counter += layer_graph[prev_node].o
            
    return output


@torch.no_grad()
def get_generator_time(model, x):
    """
    A sanity check timer function. This is only used in netadapt to get a rough idea for how fast the model is.
    """
    #    _ = model(inp)
    driving_lr = x.get('driving_lr', None)

    kp_source = model.kp_extractor(x['source'])

    if driving_lr is not None:
        kp_driving = model.kp_extractor(driving_lr)
    else:
        kp_driving = model.kp_extractor(x['driving'])

    # quick warmup
    for _ in range(10):
        generated = model.generator(x['source'],
                                    kp_source=kp_source,
                                    kp_driving=kp_driving,
                                    update_source=True,
                                    driving_lr=driving_lr)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True)

    # Just 50 round measurement
    total_time = 0
    for i in range(50):
        starter.record()
        with torch.no_grad():
            generated = model.generator(x['source'],
                                        kp_source=kp_source,
                                        kp_driving=kp_driving,
                                        update_source=True,
                                        driving_lr=driving_lr)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        total_time += curr_time

    return total_time / 50


def get_gen_input_old(model=None, x=None):
    """
    Store the input of the model after this function is run once.
    Is used for conveniently getting dummy inputs when I want to run a generator, without me needing to make sure sizes are set up correctly.
    """
    if not x is None:
        driving_lr = x.get('driving_lr', None)

        kp_source = model.kp_extractor(x['source'])

        if driving_lr is not None:
            kp_driving = model.kp_extractor(driving_lr)
        else:
            kp_driving = model.kp_extractor(x['driving'])

        get_gen_input.inputs = (x['source'], kp_source, kp_driving, True,
                                driving_lr)
    return get_gen_input.inputs


def calculate_macs(model, file_name=None):
    """
    Calculates the macs dict for the model.
    Output is some sort of dictionary with key=layer, value = macs
    """

    inputs = get_gen_input()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = profile_macs(model, inputs, reduction=None)
    return result


def total_macs(model):
    """
    Returns single number for total macs in model
    """

    macs_dict = calculate_macs(model)
    return sum(macs_dict.values())


def reverse_sort(input_list):
    """
    Input list is of the form: [(a, b), (c, d), ...]
    We want to return the list sorted by the first element of each tuple
    """
    return sorted(input_list, key=lambda x: x[0], reverse=True)


def f_set(weight, target, prune_indices):
    """
    Sets the value of a node to the weight modified with the prune indices
    """
    new_weight = torch.cat(
        [weight[:prune_indices[0]], weight[prune_indices[1]:]])
    target.set_(new_weight.contiguous())


def get_channel_reduction(deletions):
    """
    Calculates how many filters a deletion list deletes
    Used when setting out_channels for a layer.
    Example:
    get_channel_reduction([(1, 3), (5, 6)]) = 3
    """
    return sum(map(lambda x : x[1] - x[0], deletions))


@torch.no_grad()
def channel_prune(model, deletions):
    """
    Apply channel pruning to each of the conv layer in the backbone

    I've gotten rid of generic n% pruning since it bloats the code and is not even remotely close to netadapt quality.

    Deletions looks something like this
    {92: ['first', (94, 95)], 95: [(94, 95)], 93: [(94, 95)]}
    which means delete the 94 output of layer 92, the 94th input of layer 95, and the 94th input of layer 96

    The 'first' means you delete from layer 92's output instead of input signifying that it the 'first' layer whose outputs feed in as inputs to the rest of the layers.
    """
    model = copy.deepcopy(model)

    """
    Build layer_graph which graphs all the model dependencies
    """
    all_layers = [
        m for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d)
            or isinstance(m, nn.modules.batchnorm._BatchNorm))
    ]
    layer_graph = build_graph(all_layers, [
        n for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d)
            or isinstance(m, nn.modules.batchnorm._BatchNorm))
    ])

    # Deletions looks something like this
    # {92: ['first', (94, 95)], 95: [(94, 95)], 93: [(94, 95)]}
    for index, pruners in deletions.items():
        node = layer_graph[index]

        # A pruner marked with first means you delete from its outputs
        # This can only happen for convolutional layers
        if pruners[0] == 'first':
            if node.type == 'conv':
                node.value.out_channels = node.value.out_channels - get_channel_reduction(pruners[1:])
            for prune_indices in reverse_sort(pruners[1:]):
                if node.type == 'conv' and len(node.after) != 0:
                    # Prune the outupts of the convolutional layer
                    f_set(node.value.weight.detach(), node.value.weight, prune_indices)
                    if node.value.bias is not None:
                        f_set(node.value.bias.detach(), node.value.bias, prune_indices)
                else:
                    print("Should not be shrinking a batchnorm")
                # Hacky solution to modify group count for dethwise convolutions
                if node.value.groups != 1:
                    node.value.groups = node.value.weight.shape[0]

        else: # Delete from its inputs
            if node.type == 'conv':
                node.value.in_channels = node.value.in_channels - get_channel_reduction(pruners)
            for prune_indices in reverse_sort(pruners):


                # Prune inputs of a convolutional layer
                # We ignore depthwise layers since they take 1 input, their groups are just what changes.
                if node.type == 'conv' and node.value.groups == 1:
                    nvd = node.value.weight.detach()
                    nvd = torch.cat(
                        [nvd[:, :prune_indices[0]], nvd[:, prune_indices[1]:]],
                        dim=1)
                    node.value.weight.set_(nvd.contiguous())
                
                # Prune the inputs (and by definition outputs) of a batchnorm layer
                if node.type == 'bn':
                    f_set(node.value.weight.detach(), node.value.weight,
                          prune_indices)
                    f_set(node.value.bias.detach(), node.value.bias,
                          prune_indices)
                    f_set(node.value.running_mean.detach(),
                          node.value.running_mean, prune_indices)
                    f_set(node.value.running_var.detach(),
                          node.value.running_var, prune_indices)

    return model


def get_metrics_loss(metrics_dataloader, lr_size, generator_full,
                     generator_type):
    """
    Computes total loss when running generator_full on metrics_dataloader's all inputs
    """
    total_loss = 0
    with torch.no_grad():
        for y in metrics_dataloader:
            y['driving_lr'] = F.interpolate(y['driving'], lr_size)
            
            # Inputs need to be moved manually to gpu
            move_to_gpu(y)
            losses_generator, metrics_generated = generator_full(
                y, generator_type)
            loss_values = [val.mean() for val in losses_generator.values()]
            loss = sum(loss_values)
            total_loss += loss.item()

    return total_loss

def get_following_layers_skip_depthwise(following_layers, layer_graph, x):
    """
    Specialized version of follo that skips over depthwise convolved layers. Only used when sorting the outputs of a layer for importance.

    """
    following_layers.append(x.index)
    if x.type == 'conv' and x.value.groups == 1:
        return
    for y in x.after:
        get_following_layers_skip_depthwise(following_layers, layer_graph, layer_graph[y])

def follow(following_layers, layer_graph, x):
    """
    Get any possible layers which are directly impacted by editing x's outputs
    """
    following_layers.append(x.index)
    if x.type == 'conv':
        return
    for y in x.after:
        follow(following_layers, layer_graph, layer_graph[y])


def get_first_conv(layer_graph, following_layers):
    """
    Gets first convolution in following layers
    """
    for layer in following_layers:
        if layer_graph[layer].type == 'conv' and layer_graph[layer].value.groups == 1:
            return layer

def compute_deletion(layer_graph,
                     custom_deletions,
                     deleted_things,
                     layer,
                     sort,
                     custom,
                     reason=None):
    """
    Given a layer, generate the list of the indexes we need to delete from its following layers
    """
    # find all the following layers
    following_layers = []

    for after_layer in layer_graph[layer].after:
        follow(following_layers, layer_graph, layer_graph[after_layer])

    # If you delete a part of the previous one, not just all of it then you cannot just say these n are to be deleted
    # So for each layer you need to figure out what you are deleting from it, then for the next layer figure out what is deleted

    deletions = {}
    if isinstance(custom, list):
        deletions[layer] = copy.copy(custom)
    elif isinstance(custom, int):
        # The sorting magic happens here
        # Find the first convolution
        if sort:
            if layer_graph[layer].type == 'bn':
                breakpoint()
            depthwise_skipped_following_layers = []
            for after_layer in layer_graph[layer].after:
                get_following_layers_skip_depthwise(depthwise_skipped_following_layers, layer_graph, layer_graph[after_layer])

            first_conv = get_first_conv(layer_graph, depthwise_skipped_following_layers)

            if first_conv is None:
                breakpoint()

            slices = get_relevent_slice(layer_graph, first_conv, layer)
            if len(slices) == 0:
                breakpoint()

            importances = torch.zeros(layer_graph[layer].o).cuda()
            for single_slice in slices:
                importances += get_importances(layer_graph[first_conv].value.weight, single_slice)

            importances = importances.tolist()
            deletion_indices = pick_channels_with_lowest_importances(custom, importances)
            if len(deletion_indices) != custom:
                assert(False)
            deletion_list = convert_to_deletions_list(deletion_indices)


            #breakpoint()
            deletions[layer] = deletion_list
        else:
            deletions[layer] = [(layer_graph[layer].o-custom, layer_graph[layer].o)]
    else:
        assert(False)

    # If there is a plus operation, the other operand is a "mirror" of this
    # SO copy whatever we do here to the other operand
    for mirror in layer_graph[layer].mirror:
        # Start a new custom deletion for the mirror
        if reason != 'mirror':
            custom_deletions.append((mirror, copy.copy(deletions[layer]), 'mirror'))
            print("Starting a mirrorred deletion")

    # Figure out what you need to delete for the following layers
    for following_layer in following_layers:
        if following_layer in deletions:
            continue

        deletions[following_layer] = []
        counter = 0
        for previous_layer in layer_graph[following_layer].before:
            if previous_layer not in deletions:
                counter += layer_graph[previous_layer].o
                continue

            # If delete from previous layer, delete corresponding element from this layer
            for deletion in deletions[previous_layer]:
                deletions[following_layer].append(
                    (counter + deletion[0], counter + deletion[1]))
            counter += layer_graph[previous_layer].o

        # If there is another tied after layer i.e. a layer who has their output tied to the input of this layer, like in resnet, we need to delete from their outputs as well
        if len(layer_graph[following_layer].tied_after) != 0:

            total_deleted = copy.copy(deletions[following_layer])

            # Add the output layer to our work queue (custom deletions)
            for after_tied in layer_graph[following_layer].tied_after:
                if after_tied not in deleted_things:
                    deleted_things.add(after_tied)
                    custom_deletions.append((after_tied, total_deleted, 'tie'))

    # The original layer is a special case because you delete its output not input
    deletions[layer].insert(0, 'first')
    return deletions


def shrink_model(model_copy, layer_graph, layer, count, sort):
    """
    Return a new model with the layer corresponding to the argument layer shrunken by count outputs
    """
    custom_deletions = []
    deleted_things = set()
    deletions = compute_deletion(layer_graph, custom_deletions, deleted_things,
                                 layer, sort, count)
    print(deletions.keys())

    model_copy = channel_prune(model_copy, deletions)
    while len(custom_deletions) != 0:
        custom_deletion = custom_deletions[0]
        custom_deletions = custom_deletions[1:]
        deletions = compute_deletion(layer_graph, custom_deletions,
                                     deleted_things, custom_deletion[0], sort,
                                     custom_deletion[1], custom_deletion[2])
        print(deletions.keys())
        model_copy = channel_prune(model_copy, deletions)
    return model_copy

def try_reduce(curr_loss, curr_model, dataloader, layer_graph,
               layer, kp_detector, discriminator, train_params, model, target,
               current, lr_size, generator_type, metrics_dataloader,
               generator_full, sort, steps_per_it, device_ids, log_dir, discriminator_full, image_shape):
    """
    High level summary:
    Try to shrink the model to the target size by deleting layer (passed in as an argument)
    Train the model for a short term fine tune
    Test the model on the metrics dataset
    If the model beats the current best model, return it. If at any point it fails, return Nones
    This function is called ~50 times, with each input layer as an argument once in one iteration of netadapt.
    """
    with torch.no_grad():
        model_copy = copy.deepcopy(model)

    # Ignore this layer if it has only one output, or its output is never used
    if 1 >= layer_graph[layer].o:
        return None, None, None, None
    if len(layer_graph[layer].after) == 0:
        return None, None, None, None

    # Reduce the layer by size 1 to check how much it affects model size.
    model_copy = shrink_model(model_copy, layer_graph, layer, 1, sort)

    after_1_reduce = get_decode_and_bottleneck_macs(log_dir, model_copy, kp_detector, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), lr_size, image_shape, 2)
    print(after_1_reduce)
    if after_1_reduce == current:
        print("Trying to remove something that is not a part of the model")
        return None, None, None, None

    with torch.no_grad():
        model_copy = copy.deepcopy(model)
    # Calculate the number of layers that must actually be removed to hit the target
    to_remove = int((current - target) // (current - after_1_reduce))

    if to_remove == 0:
        to_remove = 1
    # Ensure the deletion is smaller than the layer size.
    if to_remove >= layer_graph[layer].o:
        print("Cannot remove enough to hit target")
        return None, None, None, None
    # Perform the actual deletion
    model_copy = shrink_model(model_copy, layer_graph, layer, to_remove, sort)
    print("done")

    # Train
    optimizer_generator = torch.optim.Adam(
        model_copy.parameters(),
        lr=train_params['lr_generator'],
        betas=(0.5, 0.999))

    with torch.no_grad():
        new_kp_detector = copy.deepcopy(kp_detector)
        new_discriminator = copy.deepcopy(discriminator)
    optimizer_kp_detector = torch.optim.Adam(new_kp_detector.parameters(), 
            lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    
    optimizer_discriminator = torch.optim.Adam(new_kp_detector.parameters(), 
            lr=train_params['lr_discriminator'], betas=(0.5, 0.999))

    generator_full.generator = model_copy
    generator_full.discriminator = new_discriminator
    generator_full.kp_extractor = new_kp_detector
    discriminator_full.kp_extractor = new_kp_detector
    discriminator_full.generator = model_copy
    discriminator_full.discriminator = new_discriminator

    counter = 0
    generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
    discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    for x in dataloader:
        break

    c = 0
    for x in tqdm(dataloader):
        c += 1
        if c > steps_per_it and steps_per_it != -1:
            break
        x['driving_lr'] = F.interpolate(x['driving'], lr_size)

        # Inputs need to manually be moved onto the gpu
        #move_to_gpu(x)
        losses_generator, generated = generator_full(x, generator_type)
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

        
    # Test the model
    total_loss = get_metrics_loss(metrics_dataloader, lr_size, generator_full,
                                  generator_type)
    print("Loss for this model is: ", total_loss)

    # Store the best module
    if curr_loss is None or total_loss < curr_loss:
        return total_loss, model_copy, new_kp_detector, new_discriminator
    else:
        return None, None, None, None


def reduce_macs(model, target, current, kp_detector, discriminator,
                train_params, dataloader, metrics_dataloader, generator_type,
                lr_size, generator_full, sort, steps_per_it, device_ids, log_dir, discriminator_full, image_shape):
    """
    Applies netadapt to reduce the model to target macs
    """

    """
    Builds a depndency graph for later use
    """
    all_layers = [
        m for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d)
            or isinstance(m, nn.modules.batchnorm._BatchNorm))
    ]
    layer_graph = build_graph(all_layers, [
        n for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d)
            or isinstance(m, nn.modules.batchnorm._BatchNorm))
    ])

    curr_model = None
    curr_kp_detector = None
    curr_loss = None
    curr_discriminator = None
    i = 0
    """
    Loops through the entire model running try_reduce on each layer
    """
    for layer in layer_graph:
        i += 1
        if layer_graph[layer].type != 'conv':
            continue
        if layer_graph[layer].is_tied:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss, t_model, t_kp_detector, t_discriminator = try_reduce(curr_loss, curr_model,
                                       dataloader, layer_graph, layer,
                                       kp_detector, discriminator,
                                       train_params, model, target, current,
                                       lr_size, generator_type,
                                       metrics_dataloader, generator_full, sort, steps_per_it, device_ids, log_dir, discriminator_full, image_shape)
        # Model returns loss != None if its model beats our current best
        if loss is not None:
            print("Updated model")
            curr_model = t_model
            curr_kp_detector = t_kp_detector
            curr_discriminator = t_discriminator
            curr_loss = loss

    if curr_model is None:
        print("Could not shrink anymore")
        raise StopIteration("Modle shrinking complete")
    print("finished netadapt iteration with loss", curr_loss)
    return curr_model, curr_kp_detector, curr_discriminator
