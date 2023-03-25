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
    
def count(name):
    obj_counter = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                obj_counter += 1
        except:
            pass 
    #print("Gc tracks ", obj_counter, "objects in ", name)

def get_sizes():
    s = {}
    obj_counter = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                size = tuple(obj.size())
                if size in s:
                    s[size] = s[size] + 1
                else:
                    s[size] = 1
                obj_counter += 1
        except:
            pass 
    return s
def plot_weight_distribution(model,
                             file_name,
                             bins=128,
                             count_nonzero_only=False):
    fig, axes = plt.subplots(8, 10, figsize=(100, 60))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if not name.endswith('weight'):
            continue
        if param.dim() > 1:
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = torch.norm(param_cpu, 'fro')
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True, alpha=0.5)
            else:
                ax.hist(param.detach().reshape(-1).cpu().numpy(),
                        bins=bins,
                        density=True,
                        color='blue',
                        alpha=0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig(file_name)


def plot_layer_parameters(model, file_name):

    fig, axes = plt.subplots(1, 1, figsize=(100, 60))
    names = [n for n, m in model.named_parameters()]

    def mul(x):
        prod = 1
        for a in x:
            prod = prod * a
        return prod

    parameter_size = [mul(m.shape) for n, m in model.named_parameters()]
    axes.set_yscale("log")
    axes.bar(names, parameter_size)
    axes.set_xticklabels(names, rotation=90)

    fig.suptitle("Number of weights")
    fig.savefig(file_name)


def plot_norm_distribution(model,
                           file_name,
                           bins=128,
                           count_nonzero_only=False):
    fig, axes = plt.subplots(8, 10, figsize=(100, 60))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if not name.endswith('weight'):
            continue
        if param.dim() > 1:
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = torch.norm(param_cpu, 'fro')
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True, alpha=0.5)
            else:
                op = []
                for i in range(param.shape[1]):
                    op.append(torch.norm(param[:, i]).detach().cpu().numpy())
                op = np.asarray(op)
                ax.hist(op, bins=bins, density=True, color='blue', alpha=0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig(file_name)


def fine_grained_prune(tensor, sparsity):
    prune.l1_unstructured(module, name='weight', amount=sparsity)


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

    def add_tie(self, node):
        self.tied_after.append(node)
    def mark_tied(self):
        self.is_tied = True
    def add_before(self, node):
        self.before.append(node)

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
                first_order_model.sync_batchnorm.SynchronizedBatchNorm2d):
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

    def add_names(names):
        index = 1
        while index < len(names):
            add(names[index - 1], names[index])
            index += 1

    # Build graph
    if os.environ.get('CONV_TYPE', 'regular') == 'regular':
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
            m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d)
    ]
    all_layers = [
        m for m in model.modules() if (isinstance(m, nn.Conv2d) or isinstance(
            m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d))
    ]

    layer_graph = build_graph(all_layers, [
        n for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d) or isinstance(
            m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d))
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


def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    """A function to calculate the number of layers to PRESERVE after pruning
    Note that preserve_rate = 1. - prune_ratio
    """
    return int(round(channels * (1 - prune_ratio)))


def select_convs(model, params):

    names = [
        m for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d) or isinstance(
            m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d))
    ]

    targets = [i for i, m in enumerate(names) if (isinstance(m, nn.Conv2d))]

    new_params = []
    for k in targets:
        new_params.append(params[k])

    return new_params


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

def calc_latencies(model):
    all_layers = [
        m for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d) or isinstance(
            m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d))
    ]
    layer_graph = build_graph(all_layers, [
        n for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d) or isinstance(
            m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d))
    ])
    starting_shape = (512, 512)

    def convert_shape(shape, layer):
        if layer.type == 'conv':
            f = layer.value.kernel_size
            s = layer.value.stride
            new_shape = [0, 0]
            new_shape[0] = (int((shape[0] - f[0]) / s[0])) + 1
            new_shape[1] = (int((shape[1] - f[1]) / s[1])) + 1
            return tuple(new_shape)
        return shape

    def calc_flops(out_shape, node):
        if node.type == 'conv':
            return (node.value.kernel_size[0]**2) * node.value.weight.shape[
                0] * node.value.weight.shape[1] * out_shape[0] * out_shape[1]
        else:
            return node.o

    def calc_time(in_shape, node):
        if node.type == 'conv':
            temp_input_cpu = torch.rand(1, node.value.weight.shape[1],
                                        in_shape[0], in_shape[1])
        else:
            temp_input_cpu = torch.rand(1, node.value.weight.shape[0],
                                        in_shape[0], in_shape[1])
        temp_input = temp_input_cpu.cuda()

        for _ in range(10):
            generated = node.value(temp_input)        
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        total_time = 0
        for i in range(50):
            starter.record()
            generated = node.value(temp_input)        
            ender.record()

            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            total_time += curr_time
        return total_time/50


    def follow(node, results, shape, times):
        if node.index in times:
            return
        #print("follow node ", node.index)
        new_shape_ = convert_shape(shape, node)
        with torch.no_grad():
            results[node.index] = calc_time(new_shape_, node)


        for node_index in node.after:
            follow(layer_graph[node_index], results, new_shape_, times)

    t_results = {}
    times = {}
    for i in [0, 24, 25, 26, 28, 30]:
        follow(layer_graph[i], t_results, starting_shape, times)

    fig, axes = plt.subplots(1, 1, figsize=(100, 100))
    names = [
        n for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d) or isinstance(
            m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d))
    ]
    parameter_size = [t_results[i] for i in range(len(names))]
    parameter_size = select_convs(model, parameter_size)
    axes.set_yscale("log")
    names = select_convs(model, names)
    axes.bar(names, parameter_size)
    axes.set_xticklabels(names, rotation=90)

    fig.suptitle("Times plot")
    file_name = 'latencies'
    fig.savefig(file_name)
    return t_results

def total_latency(model):
    macs_dict = calc_latencies(model)
    s = 0
    for k in macs_dict:
        s+=macs_dict[k]
    return s


@torch.no_grad()
def channel_prune(model, prune_ratio, deletions=None):
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
            m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d))
    ]
    layer_graph = build_graph(all_layers, [
        n for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d) or isinstance(
            m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d))
    ])

    if deletions is None:
        for base_index in layer_graph:
            node = layer_graph[base_index]
            total_rows = 0
            counter = 0
            pruners = []
            curr_index = 0
            for node_index in node.before:
                total_rows += int(prune_ratio * layer_graph[node_index].o)

                pruners.append(
                    (int(counter + (layer_graph[node_index].o * prune_ratio)),
                     counter + (layer_graph[node_index].o)))
                counter += layer_graph[node_index].o

            to_delete = total_rows

            for i in range(len(pruners)):
                prune_indices = pruners[len(pruners) - i - 1]

                if node.type == 'conv':
                    nvd = node.value.weight.detach()
                    nvd = torch.cat(
                        [nvd[:, :prune_indices[0]], nvd[:, prune_indices[1]:]],
                        dim=1)
                    node.value.weight.set_(nvd.clone().detach())

                if node.type == 'bn':

                    def f_set(nvd, w, prune_indices):
                        nvd = torch.cat(
                            [nvd[:prune_indices[0]], nvd[prune_indices[1]:]])
                        w.set_(nvd.clone().detach())

                    f_set(node.value.weight.detach(), node.value.weight,
                          prune_indices)
                    f_set(node.value.bias.detach(), node.value.bias,
                          prune_indices)
                    f_set(node.value.running_mean.detach(),
                          node.value.running_mean, prune_indices)
                    f_set(node.value.running_var.detach(),
                          node.value.running_var, prune_indices)
            op_delete = int((prune_ratio) * node.o)
            if node.type == 'conv':
                if len(node.after) != 0:
                    # Prune the outupts
                    node.value.weight.set_(
                        node.value.weight.detach()[:op_delete])
                    node.value.bias.set_(node.value.bias.detach()[:op_delete])
        return model
    else:
        for index, pruners in deletions.items():
            node = layer_graph[index]
            if pruners[0] == 'first':
                if node.type == 'conv' and len(node.after) != 0:
                    # Prune the outupts
                    node.value.weight.set_(
                        node.value.weight.detach()[:pruners[1][0]].contiguous())
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

        #for base_index in layer_graph:
        #    node = layer_graph[base_index]
        #    total_rows = 0
        #    counter = 0
        #    pruners = []
        #    curr_index = 0
        #    for node_index in node.before:
        #        total_rows += int(prune_ratios[node_index] *
        #                          layer_graph[node_index].o)
        #        pruners.append((int(counter + (layer_graph[node_index].o *
        #                                       prune_ratios[node_index])),
        #                        counter + (layer_graph[node_index].o)))
        #        counter += layer_graph[node_index].o

        #    to_delete = total_rows

        return model


@torch.no_grad()
def sensitivity_scan(model,
                     dataloader,
                     scan_step=0.1,
                     scan_start=0.4,
                     scan_end=1.0,
                     verbose=True):
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    named_conv_weights = [(name, param) for (name, param) \
                          in model.named_parameters() if param.dim() > 1]
    for i_layer, (name, param) in enumerate(named_conv_weights):
        param_clone = param.detach().clone()
        accuracy = []
        for sparsity in tqdm(
                sparsities,
                desc=
                f'scanning {i_layer}/{len(named_conv_weights)} weight - {name}'
        ):
            prune.l1_unstructured(param, name='weight', amount=sparsity)
            fine_grained_prune(param.detach(), sparsity=sparsity)
            acc = evaluate(model, dataloader, verbose=False)
            if verbose:
                print(f'\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%',
                      end='')
            # restore
            param.copy_(param_clone)
            accuracy.append(acc)
        if verbose:
            print(
                f'\r    sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]',
                end='')
        accuracies.append(accuracy)
    return sparsities, accuracies

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

def try_reduce(curr_loss, curr_model, per_layer_macs, dataloader, layer_graph, layer, kp_detector, discriminator, train_params, model, target, current, lr_size, generator_type, metrics_dataloader, generator_full):
    custom_deletions = []
    deleted_things = set()
    def compute_deletion(layer, custom=None):
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

    
        def edit_macs(following_layers, layer, k):
            temp = current
            actual_layer = None
            for t_layer in layer_graph:
                if layer_graph[t_layer].index == layer:
                    actual_layer = layer_graph[t_layer]
            temp -= per_layer_macs[layer] * k / actual_layer.o
            for temp_layer in following_layers:
                actual_layer = None
                for t_layer in layer_graph:
                    if layer_graph[t_layer].index == temp_layer:
                        actual_layer = layer_graph[t_layer]
                temp -= per_layer_macs[temp_layer] * k / actual_layer.i

            return temp

        i = 0
        if custom is None:
            fail = False
            while edit_macs(following_layers, layer, i) > target:
                if layer_graph[layer].o > i:
                    i += 1
                else:
                    fail = True
                    break

            if fail:
                return None

            else:
                pass
        # Build the pruning graph
        prune_ratios = {}
        prune_ratios[layer_graph[layer].index] = i / layer_graph[layer].o

        # If you delete a part of the previous one, not just all of it then you cannot just say these n are to be deleted
        # So for each layer you need to figure out what you are deleting from it, then for the next layer figure out what is deleted

        # So make a map of deletions
        deletions = {}
        if custom is None:
            deletions[layer] = [(layer_graph[layer].o - i, layer_graph[layer].o)]
        else:
            deletions[layer] = [(layer_graph[layer].o - custom, layer_graph[layer].o)]


        for following_layer in following_layers:
            if following_layer in deletions:
                continue

            deletions[following_layer] = []
            counter = 0
            for previous_layer in layer_graph[following_layer].before:
                if previous_layer not in deletions:
                    continue
                for deletion in deletions[previous_layer]:
                    deletions[following_layer].append(
                        (counter + deletion[0], counter + deletion[1]))
                counter += layer_graph[previous_layer].o

            if len(layer_graph[following_layer].tied_after) != 0:
                total_deleted = 0
                for del_tuple in deletions[following_layer]:
                    total_deleted += del_tuple[1] - del_tuple[0]
                for after_tied in layer_graph[following_layer].tied_after:
                    if after_tied not in deleted_things:
                        deleted_things.add(after_tied)
                        custom_deletions.append((after_tied, total_deleted))
                


        # The original layer is a special case because you delete its output not input
        deletions[layer].insert(0, 'first')
        return deletions

    model_copy = copy.deepcopy(model)
    if 1 >= layer_graph[layer].o:
        return None, None
    deletions = compute_deletion(layer, 1)
    print(deletions)
    if deletions is None:
        return None, None
    model_copy = channel_prune(
        model_copy, 0.1,
        deletions)  # The 0.1 is a dummy variable for now, gets ignored
    while len(custom_deletions) != 0:
        custom_deletion =  custom_deletions[0]
        custom_deletions = custom_deletions[1:]
        deletions = compute_deletion(custom_deletion[0], custom_deletion[1])
        model_copy = channel_prune(
            model_copy, 0.1,
            deletions)  # The 0.1 is a dummy variable for now, gets ignored

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
        deletions = compute_deletion(layer, to_remove)
        if deletions is None:
            return None, None
        model_copy = channel_prune(
            model_copy, 0.1,
            deletions)  # The 0.1 is a dummy variable for now, gets ignored
        print(deletions)
        while len(custom_deletions) != 0:
            custom_deletion =  custom_deletions[0]
            custom_deletions = custom_deletions[1:]
            deletions = compute_deletion(custom_deletion[0], custom_deletion[1])
            print(deletions)
            model_copy = channel_prune(
                model_copy, 0.1,
                deletions)  # The 0.1 is a dummy variable for now, gets ignored


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
        #print("Before")
        #print(get_sizes())
        #count("before gen")
        losses_generator, generated = generator_full(x, generator_type)
        #print("After")
        #print(get_sizes())
        #count("after gen")
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
            m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d))
    ]
    layer_graph = build_graph(all_layers, [
        n for n, m in model.named_modules()
        if (isinstance(m, nn.Conv2d) or isinstance(
            m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d))
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
