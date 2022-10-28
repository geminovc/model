from tqdm import trange
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
def plot_weight_distribution(model, file_name, bins=128, count_nonzero_only=False):
    fig, axes = plt.subplots(8,10, figsize=(100, 60))
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
                ax.hist(param_cpu, bins=bins, density=True, 
                        alpha = 0.5)
            else:
                ax.hist(param.detach().reshape(-1).cpu().numpy(), bins=bins, density=True, 
                        color = 'blue', alpha = 0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig(file_name)

def plot_layer_parameters(model, file_name):

    fig, axes = plt.subplots(1, 1, figsize=(100,60))
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

def plot_norm_distribution(model, file_name, bins=128, count_nonzero_only=False):
    fig, axes = plt.subplots(8,10, figsize=(100, 60))
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
                ax.hist(param_cpu, bins=bins, density=True, 
                        alpha = 0.5)
            else:
                op = []
                for i in range(param.shape[1]):
                    op.append(torch.norm(param[:, i]).detach().cpu().numpy())
                op = np.asarray(op)
                ax.hist(op, bins=bins, density=True, 
                        color = 'blue', alpha = 0.5)
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
        self.after = set()
    def add_before(self, node):
        self.before.append(node)
    def add_after(self, node):
        self.after.add(node)

def build_graph(all_layers, names):
    # For the sake of getting this working we are going to hardcode each layer
    graph = {}
    for index in range(len(all_layers)):
        if isinstance(all_layers[index], nn.Conv2d):

            graph[index] = Node(index, 'conv', all_layers[index].in_channels, all_layers[index].out_channels, all_layers[index])
        elif isinstance(all_layers[index], first_order_model.sync_batchnorm.SynchronizedBatchNorm2d):
            graph[index] = Node(index, 'bn', all_layers[index].num_features, all_layers[index].num_features, all_layers[index])
        else:
            graph[index] = Node(index)

    def get_index(name):
        return names.index(name)

    def add(name1, name2):
        index1 = get_index(name1)
        index2 = get_index(name2)
        graph[index1].add_after(index2)
        graph[index2].add_before(index1)

    def add_names(names):
        index = 1
        while index < len(names):
            add(names[index-1], names[index])
            index += 1

    # Build graph
    add_names( ['dense_motion_network.hourglass.encoder.down_blocks.0.conv', 'dense_motion_network.hourglass.encoder.down_blocks.0.norm', 'dense_motion_network.hourglass.encoder.down_blocks.1.conv', 'dense_motion_network.hourglass.encoder.down_blocks.1.norm', 'dense_motion_network.hourglass.encoder.down_blocks.2.conv', 'dense_motion_network.hourglass.encoder.down_blocks.2.norm', 'dense_motion_network.hourglass.encoder.down_blocks.3.conv', 'dense_motion_network.hourglass.encoder.down_blocks.3.norm', 'dense_motion_network.hourglass.encoder.down_blocks.4.conv', 'dense_motion_network.hourglass.encoder.down_blocks.4.norm', 'dense_motion_network.hourglass.decoder.up_blocks.0.conv', 'dense_motion_network.hourglass.decoder.up_blocks.0.norm', 'dense_motion_network.hourglass.decoder.up_blocks.1.conv', 'dense_motion_network.hourglass.decoder.up_blocks.1.norm', 'dense_motion_network.hourglass.decoder.up_blocks.2.conv', 'dense_motion_network.hourglass.decoder.up_blocks.2.norm', 'dense_motion_network.hourglass.decoder.up_blocks.3.conv', 'dense_motion_network.hourglass.decoder.up_blocks.3.norm', 'dense_motion_network.hourglass.decoder.up_blocks.4.conv', 'dense_motion_network.hourglass.decoder.up_blocks.4.norm'] )

    # Add the dense motion skip connections
    add('dense_motion_network.hourglass.encoder.down_blocks.3.norm', 'dense_motion_network.hourglass.decoder.up_blocks.1.conv')
    add('dense_motion_network.hourglass.encoder.down_blocks.2.norm', 'dense_motion_network.hourglass.decoder.up_blocks.2.conv')
    add('dense_motion_network.hourglass.encoder.down_blocks.1.norm', 'dense_motion_network.hourglass.decoder.up_blocks.3.conv')
    add('dense_motion_network.hourglass.encoder.down_blocks.0.norm', 'dense_motion_network.hourglass.decoder.up_blocks.4.conv')

    # Add the dense motion outputs partly (First part)
    add('dense_motion_network.hourglass.decoder.up_blocks.4.norm', 'dense_motion_network.mask')
    add('dense_motion_network.hourglass.decoder.up_blocks.4.norm', 'dense_motion_network.occlusion')
    add('dense_motion_network.hourglass.decoder.up_blocks.4.norm', 'dense_motion_network.lr_occlusion')
    add('dense_motion_network.hourglass.decoder.up_blocks.4.norm', 'dense_motion_network.hr_background_occlusion')

    add('lr_first.conv', 'lr_first.norm')
    add_names(['hr_first.conv', 'hr_first.norm', 'hr_down_blocks.0.conv', 'hr_down_blocks.0.norm', 'hr_down_blocks.1.conv', 'hr_down_blocks.1.norm'])
    add_names(['first.conv', 'first.norm', 'down_blocks.0.conv', 'down_blocks.0.norm', 'down_blocks.1.conv', 'down_blocks.1.norm', 'bottleneck.r0.norm1', 'bottleneck.r0.conv1', 'bottleneck.r0.norm2', 'bottleneck.r0.conv2', 'bottleneck.r1.norm1', 'bottleneck.r1.conv1', 'bottleneck.r1.norm2', 'bottleneck.r1.conv2', 'bottleneck.r2.norm1', 'bottleneck.r2.conv1', 'bottleneck.r2.norm2', 'bottleneck.r2.conv2', 'bottleneck.r3.norm1', 'bottleneck.r3.conv1', 'bottleneck.r3.norm2', 'bottleneck.r3.conv2', 'bottleneck.r4.norm1', 'bottleneck.r4.conv1', 'bottleneck.r4.norm2', 'bottleneck.r4.conv2', 'bottleneck.r5.norm1', 'bottleneck.r5.conv1', 'bottleneck.r5.norm2', 'bottleneck.r5.conv2', 'up_blocks.0.conv','up_blocks.0.norm', 'up_blocks.1.conv', 'up_blocks.1.norm', 'hr_up_blocks.0.conv', 'hr_up_blocks.0.norm','hr_up_blocks.1.conv','hr_up_blocks.1.norm' , 'final'])

    # Features get concatted into down block
    add('down_blocks.1.norm', 'bottleneck.r0.norm1')

    # Second up block has 32 lr features added
    add('lr_first.norm', 'up_blocks.1.conv')

    # Add 2x hr down outputs to first hr up
    add('hr_down_blocks.1.norm', 'hr_up_blocks.0.conv')
    add('hr_down_blocks.1.norm', 'hr_up_blocks.0.conv')

    add('hr_down_blocks.0.norm', 'hr_up_blocks.1.conv')
    add('hr_down_blocks.0.norm', 'hr_up_blocks.1.conv')

    return graph

@torch.no_grad()
def apply_channel_sorting(model):
    model = copy.deepcopy(model)  # do not modify the original model
    # fetch all the conv and bn layers from the backbone
    all_convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]

    all_convs = all_convs[:10] + all_convs[14:]
    all_bns = [m for m in model.modules() if isinstance(m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d)]
    all_layers = [m for m in model.modules() if (isinstance(m, nn.Conv2d) or isinstance(m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d))]

    layer_graph = build_graph(all_layers)

    def is_valid_helper(node, layer_graph):
        if len(node.after) > 1 or len(node.after) == 0:
            return False, None
        if node.type == 'conv':
            return True, node
        return is_valid_helper(node.after[0])

    def is_valid(node, layer_graph):
        if len(node.after) > 1 or len(node.after) == 0:
            return False, None
        return is_valid_helper(node.after[0])

    for node in layer_graph:
        if node.type == 'conv':
            valid, end = is_valid(node, layer_graph)
            if not valid:
                continue

            importance = get_input_channel_importance(layer_graph[end.index].weight)
            # sorting from large to small
            sort_idx = torch.argsort(importance, descending=True) 

            temp = node
            layer_graph[temp.index].weight.copy_(torch.index_select(
                layer_graph[temp.index].weight.detach(), 0, sort_idx))
            layer_graph[temp.index].bias.copy_(torch.index_select(
                layer_graph[temp.index].bias.detach(), 0, sort_idx))
            temp=temp.after[0]

            while temp.type != 'conv':

                bn = layer_graph[temp.index]
                for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
                    tensor_to_apply = getattr(bn, tensor_name)
                    tensor_to_apply.copy_(
                        torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
                    )
                temp = temp.after[0]

            layer_graph[temp.index].weight.copy_(torch.index_select(
                layer_graph[temp.index].weight.detach(), 1, sort_idx))
    return model

def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    """A function to calculate the number of layers to PRESERVE after pruning
    Note that preserve_rate = 1. - prune_ratio
    """
    return int(round(channels * (1-prune_ratio)))

@torch.no_grad()
def channel_prune(model, prune_ratio):
    """Apply channel pruning to each of the conv layer in the backbone
    Note that for prune_ratio, we can either provide a floating-point number,
    indicating that we use a uniform pruning rate for all layers, or a list of
    numbers to indicate per-layer pruning rate.
    """
    print("START")
    # sanity check of provided prune_ratio

    all_layers = [m for n, m in model.named_modules() if (isinstance(m, nn.Conv2d) or isinstance(m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d))]
    layer_graph = build_graph(all_layers, [n for n, m in model.named_modules() if (isinstance(m, nn.Conv2d) or isinstance(m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d))])


    for base_index in layer_graph:
        node = layer_graph[base_index]
        total_rows = 0
        counter = 0
        for node_index in node.before:
            total_rows += int(prune_ratio * layer_graph[node_index].o)
            counter += layer_graph[node_index].o

        to_delete = total_rows
        if counter != node.i:
            print(to_delete, node.i, counter, base_index, node.before)
        # Prune the inputs
        if node.type == 'conv':
            node.value.weight.set_(node.value.weight.detach()[:,to_delete:])
            print(node.value.weight.shape)

        if node.type == 'bn':
            node.value.weight.set_(node.value.weight.detach()[to_delete:])
            node.value.bias.set_(node.value.bias.detach()[to_delete:])
            node.value.running_mean.set_(node.value.running_mean.detach()[to_delete:])
            node.value.running_var.set_(node.value.running_var.detach()[to_delete:])
            
        op_delete =int(prune_ratio* node.o)
        if  node.type == 'conv':
            if len(node.after) != 0:
                # Prune the outupts
                node.value.weight.set_(node.value.weight.detach()[op_delete:])
                node.value.bias.set_(node.value.bias.detach()[op_delete:])


    print("end")
    return model


@torch.no_grad()
def sensitivity_scan(model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True):
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    named_conv_weights = [(name, param) for (name, param) \
                          in model.named_parameters() if param.dim() > 1]
    for i_layer, (name, param) in enumerate(named_conv_weights):
        param_clone = param.detach().clone()
        accuracy = []
        for sparsity in tqdm(sparsities, desc=f'scanning {i_layer}/{len(named_conv_weights)} weight - {name}'):
            prune.l1_unstructured(param, name='weight', amount=sparsity)
            fine_grained_prune(param.detach(), sparsity=sparsity)
            acc = evaluate(model, dataloader, verbose=False)
            if verbose:
                print(f'\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%', end='')
            # restore
            param.copy_(param_clone)
            accuracy.append(acc)
        if verbose:
            print(f'\r    sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]', end='')
        accuracies.append(accuracy)
    return sparsities, accuracies

def get_frame_from_video_codec(frame_tensor, nr_list, dr_list, quantizer, bitrate):
    """ go through the encoder/decoder pipeline to get a 
        representative decoded frame
    """
    # encode every frame as a keyframe with new encoder/decoder
    frame_data = frame_tensor.data.cpu().numpy()
    decoded_data = np.zeros(frame_data.shape, dtype=np.uint8) 
    
    frame_data = np.transpose(frame_data, [0, 2, 3, 1])
    frame_data *= 255
    frame_data = frame_data.astype(np.uint8)
    
    nr_list = nr_list.data.cpu().numpy()
    dr_list = dr_list.data.cpu().numpy()

    for i, (frame, nr, dr) in enumerate(zip(frame_data, nr_list, dr_list)):
        av_frame = av.VideoFrame.from_ndarray(frame)
        av_frame.pts = 0
        av_frame.time_base = Fraction(nr, dr)
        encoder, decoder = Vp8Encoder(), Vp8Decoder()
        if bitrate == None:
            payloads, timestamp = encoder.encode(av_frame, quantizer=quantizer, enable_gcc=False)
        else:
            payloads, timestamp = encoder.encode(av_frame, quantizer=-1, \
                    target_bitrate=bitrate, enable_gcc=False)
        payload_data = [vp8_depayload(p) for p in payloads]
        
        jitter_frame = JitterFrame(data=b"".join(payload_data), timestamp=timestamp)
        decoded_frames = decoder.decode(jitter_frame)
        decoded_frame = decoded_frames[0].to_rgb().to_ndarray()
        decoded_data[i] = np.transpose(decoded_frame, [2, 0, 1]).astype(np.uint8)
    return torch.from_numpy(img_as_float32(decoded_data))


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


    
    plot_norm_distribution(generator, 'before_load_norm.png')
    plot_weight_distribution(generator, 'before_load.png')
    plot_layer_parameters(generator, 'before_load_params.png')
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

    plot_norm_distribution(generator, 'after_load_norm.png')
    plot_weight_distribution(generator, 'after_load.png')
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
    #generator = apply_channel_sorting(generator)
    generator = channel_prune(generator, 0.5)

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)
    
    vgg_model = Vgg19()
    original_lpips = lpips.LPIPS(net='vgg')
    vgg_face_model = VggFace16()

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)
        original_lpips = original_lpips.cuda()
        vgg_model = vgg_model.cuda()
        vgg_face_model = vgg_face_model.cuda()
    
    loss_fn_vgg = vgg_model.compute_loss
    face_lpips = vgg_face_model.compute_loss

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                print(x['driving'].shape, "input shape")
                if use_lr_video or use_RIFE:
                    lr_frame = F.interpolate(x['driving'], lr_size)
                    if train_params.get('encode_video_for_training', False):
                        target_bitrate = train_params.get('target_bitrate', None)
                        quantizer_level = train_params.get('quantizer_level', -1)
                        if target_bitrate == 'random':
                            target_bitrate = np.random.randint(15, 75) * 1000
                        nr = x.get('time_base_nr', torch.ones(lr_frame.size(dim=0), dtype=int))
                        dr = x.get('time_base_dr', 30000 * torch.ones(lr_frame.size(dim=0), dtype=int))
                        x['driving_lr'] = get_frame_from_video_codec(lr_frame, nr, \
                                dr, quantizer_level, target_bitrate) 
                    else:
                        x['driving_lr'] = lr_frame

                losses_generator, generated = generator_full(x, generator_type)

                if epoch == 0 or True:
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

            plot_weight_distribution(generator, 'unpruned.png')

            for name, module in generator.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=0.8)
                    prune.remove(module, name='weight')
                # prune 40% of connections in all linear layers
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.8)
                    prune.remove(module, name='weight')
                else:
                    pass
            plot_weight_distribution(generator, 'pruned.png')
            inputs = []
            generator_full.eval()
            generator_full(x,generator_type, inputs)
            i = 0
            with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_stack=True) as prof:
                generator(inputs[0], **inputs[1])
            prof.export_stacks("profiler_stacks.txt", "self_cpu_time_total")
            print("DONE PROFILING")

            


            inputs[0] = inputs[0].detach()
            for key in inputs[1].keys():
                if type(inputs[1][key]) is dict:
                    for key_ in inputs[1][key].keys():
                        inputs[1][key][key_] = inputs[1][key][key_].detach()

            all_layers = [m for m in generator.modules() if (isinstance(m, nn.Conv2d) or isinstance(m, first_order_model.sync_batchnorm.SynchronizedBatchNorm2d))]
            print(len(all_layers))
            print("len^")
            positional_inputs = (inputs[0], inputs[1]['kp_driving'], inputs[1]['kp_source'])
            breakpoint()
            generator.eval()
            with torch.no_grad():
                torch.onnx.export(generator, tuple(inputs), 'log/trial1.onnx', export_params=True, opset_version=16)
            print("SAVED")


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
