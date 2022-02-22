import sys
sys.path.append("../..")
from keypoint_based_face_models import KeypointBasedFaceModels
import torch
from torch import nn
import torch.nn.functional as F
# from mmcv.ops.point_sample import bilinear_grid_sample
import yaml
from skimage import img_as_float32
import imageio 
import time
import os, sys
from skimage.draw import circle
import matplotlib.pyplot as plt
import collections
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
import queue
import threading
from torch.nn import Conv2d
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import trange
from first_order_model.sync_batchnorm import DataParallelWithCallback
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
from first_order_model.modules.discriminator import MultiScaleDiscriminator
from first_order_model.frames_dataset import FramesDataset, DatasetRepeater
from torchvision import models
import numpy as np
from torch.autograd import grad
from first_order_model.logger import Logger, Visualizer
from first_order_model.modules.util import kp2gaussian, make_coordinate_grid, AntiAliasInterpolation2d
from first_order_model.modules.model import Vgg19, ImagePyramide, Transform, DiscriminatorFullModel, detach_kp
from first_order_model.quantization.utils import quantize_model, print_average_and_std, get_params, get_coder_modules_to_fuse, QUANT_ENGINE


USE_FAST_CONV2 = False
USE_FLOAT_16 = False
USE_QUANTIZATION = False
USE_CUDA = False
IMAGE_RESOLUTION = 256
NUM_RUNS = 1000


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def forward(self, x):
        kp_source = self.kp_extractor(x['source'])
        kp_driving = self.kp_extractor(x['driving'])

        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            # TODO
            # if self.loss_weights['equivariance_jacobian'] != 0:
            #     import pdb
            #     pdb.set_trace()
            #     transformed_kp_v_with_grad = torch.autograd.Variable(transformed_kp['value'], requires_grad=True)
            #     jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']), transformed_kp['jacobian'])
            #     torch.matmul(transform.jacobian(transformed_kp_v_with_grad), transformed_kp['jacobian'])

            #     normed_driving = torch.inverse(kp_driving['jacobian'])
            #     normed_transformed = jacobian_transformed
            #     value = torch.matmul(normed_driving, normed_transformed)

            #     eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

            #     value = torch.abs(eye - value).mean()
            #     loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        return loss_values, generated


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        if not USE_FAST_CONV2:
            self.conv1 = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                                padding=padding)
            self.conv2 = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                            padding=padding)
        else:
            self.conv1 = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, stride=(1, 1), padding= padding, groups=in_features)
            self.point_conv1 = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=(1, 1))
            self.conv2 = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, stride=(1, 1), padding= padding, groups=in_features)
            self.point_conv2 = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=(1, 1))
        self.norm1 = nn.BatchNorm2d(in_features, affine=True)
        self.norm2 = nn.BatchNorm2d(in_features, affine=True)
        self.relu = torch.nn.ReLU()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()


    def forward(self, x):
        x = self.quant(x)
        out = self.norm1(x)
        out = self.relu(out)
        if not USE_FAST_CONV2:
            out = self.conv1(out)
        else:
            out = self.conv1(out)
            out = self.point_conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        if not USE_FAST_CONV2:
            out = self.conv2(out)
        else:
            out = self.conv2(out)
            out = self.point_conv2(out)
        out = self.dequant(out)
        x = self.dequant(x)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()
        if not USE_FAST_CONV2:
            self.conv = Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                                padding=padding, groups=groups)
        else:
            self.depth_conv = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, stride=(1, 1), padding= padding, groups=in_features)
            self.conv = Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=(1, 1))
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.relu = torch.nn.ReLU()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        out = F.interpolate(x, scale_factor=2)
        if not USE_FAST_CONV2:
            out = self.conv(out)
        else:
            out = self.depth_conv(out)
            out = self.conv(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dequant(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        if not USE_FAST_CONV2:
            self.conv = Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                                padding=padding, groups=groups)
        else:
            self.depth_conv = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, stride=(1, 1), padding= padding, groups=in_features)
            self.conv = Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=(1, 1))
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        self.relu = torch.nn.ReLU()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        if not USE_FAST_CONV2:
            out = self.conv(x)
        else:
            out = self.depth_conv(x)
            out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.dequant(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        if not USE_FAST_CONV2:
            self.conv = Conv2d(in_channels=in_features, out_channels=out_features,
                                kernel_size=kernel_size, padding=padding, groups=groups)
        else:
            self.depth_conv = Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, stride=(1, 1), padding= padding, groups=in_features)
            self.conv = Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=(1, 1))
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.relu = torch.nn.ReLU()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        if not USE_FAST_CONV2:
            out = self.conv(x)
        else:
            out = self.depth_conv(x)
            out = self.conv(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dequant(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        # x = self.quant(x)
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        # outs = self.dequant(outs)
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()
        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None,
                 estimate_jacobian=False, for_onnx=False):
        super(OcclusionAwareGenerator, self).__init__()
        if dense_motion_params is not None:
            if for_onnx:
                self.dense_motion_network = DenseMotionNetwork_ONNX(num_kp=num_kp, num_channels=num_channels,
                                                            estimate_occlusion_map=estimate_occlusion_map,
                                                            **dense_motion_params, for_onnx=True)
            else:
                self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                            estimate_occlusion_map=estimate_occlusion_map,
                                                            **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
        self.for_onnx = for_onnx
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        if self.for_onnx:
            return bilinear_grid_sample(inp, deformation), deformation
        else:
            return F.grid_sample(inp, deformation), deformation

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out, _ = self.deform_input(out, deformation)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = self.dequant(out)
                occlusion_map = self.dequant(occlusion_map)
                out = out * occlusion_map


            output_dict["deformed"], deformation = self.deform_input(source_image, deformation)
            output_dict["deformation"] = deformation

        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.quant(out)
        out = self.final(out)
        out = self.dequant(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict


class OcclusionAwareGenerator_with_time(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None,
                 estimate_jacobian=False, for_onnx=False):
        super(OcclusionAwareGenerator_with_time, self).__init__()
        if dense_motion_params is not None:
            if for_onnx:
                self.dense_motion_network = DenseMotionNetwork_ONNX(num_kp=num_kp, num_channels=num_channels,
                                                            estimate_occlusion_map=estimate_occlusion_map,
                                                            **dense_motion_params, for_onnx=True)
            else:
                self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                            estimate_occlusion_map=estimate_occlusion_map,
                                                            **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
        self.for_onnx = for_onnx
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        if self.for_onnx:
            return bilinear_grid_sample(inp, deformation), deformation
        else:
            return F.grid_sample(inp, deformation), deformation

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        start_time = time.time()
        out = self.first(source_image)
        first_time = time.time() - start_time
        down_blocks_time = []
        for i in range(len(self.down_blocks)):
            start_time = time.time()
            out = self.down_blocks[i](out)
            down_blocks_time.append(time.time() - start_time)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        dense_morion_time = 0
        deform_time = 0
        if self.dense_motion_network is not None:
            start_time = time.time()
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            dense_morion_time = time.time() - start_time
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            start_time = time.time()
            out, _ = self.deform_input(out, deformation)
            deform_time = time.time() - start_time

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = self.dequant(out)
                occlusion_map = self.dequant(occlusion_map)
                out = out * occlusion_map

            output_dict["deformed"], deformation = self.deform_input(source_image, deformation)
            output_dict["deformation"] = deformation

        # Decoding part
        start_time = time.time()
        out = self.bottleneck(out)
        bottleneck_time = time.time() - start_time
        up_blocks_time = []
        for i in range(len(self.up_blocks)):
            start_time = time.time()
            out = self.up_blocks[i](out)
            up_blocks_time.append(time.time() - start_time)
        out = self.quant(out)
        start_time = time.time()
        out = self.final(out)
        final_time = time.time() - start_time
        out = self.dequant(out)
        start_time = time.time()
        out = F.sigmoid(out)
        sigmoid_time = time.time() - start_time

        output_dict["prediction"] = out

        return output_dict, first_time, down_blocks_time, dense_morion_time, deform_time, bottleneck_time, up_blocks_time, final_time, sigmoid_time


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01, for_onnx=False):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 1) * (num_channels + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        self.mask = Conv2d(self.hourglass.out_filters, num_kp + 1, kernel_size=(7, 7), padding=(3, 3))

        if estimate_occlusion_map:
            self.occlusion = Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        
        self.for_onnx = for_onnx
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving['value'], spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source['value'], spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        #adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        # zeros = zeros.to(f'cuda:{heatmap.get_device()}')
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        # identity_grid = identity_grid.to(f'cuda:{source_image.get_device()}')
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)
        # TODO
        if 'jacobian' in kp_driving and not self.for_onnx:
            inversed = torch.inverse(self.dequant(kp_driving['jacobian']))
            jacobian = torch.matmul(self.dequant(kp_source['jacobian']), inversed)
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 2)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        if self.for_onnx:
            sparse_deformed = bilinear_grid_sample(source_repeat, sparse_motions)
        else:
            sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        out_dict['sparse_deformed'] = deformed_source

        input = torch.cat([heatmap_representation, deformed_source], dim=2)
        input = input.view(bs, -1, h, w)

        prediction = self.hourglass(input)
        prediction = self.quant(prediction)
        mask = self.mask(prediction)
        mask = self.dequant(mask)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict['deformation'] = deformation

        # Sec. 3.2 in the paper
        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict


class DenseMotionNetwork_with_time(DenseMotionNetwork):
    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        start_time = time.time()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        heatmap_representation_time = time.time() - start_time
        start_time = time.time()
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)
        sparse_motion_time = time.time() - start_time
        start_time = time.time()
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        create_deformed_source_time = time.time() - start_time
        out_dict['sparse_deformed'] = deformed_source

        input = torch.cat([heatmap_representation, deformed_source], dim=2)
        input = input.view(bs, -1, h, w)
        
        start_time = time.time()
        prediction = self.hourglass(input)
        hourglass_time = time.time() - start_time
        prediction = self.quant(prediction)
        start_time = time.time()
        mask = self.mask(prediction)
        mask_time = time.time() - start_time
        mask = self.dequant(mask)
        start_time = time.time()
        mask = F.softmax(mask, dim=1)
        softmax_time = time.time() - start_time
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        start_time = time.time()
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)
        deformation_time = time.time() - start_time

        out_dict['deformation'] = deformation

        # Sec. 3.2 in the paper
        occlusion_time = 0
        if self.occlusion:
            start_time = time.time()
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map
            occlusion_time = time.time() - start_time

        return out_dict, heatmap_representation_time, sparse_motion_time, create_deformed_source_time, \
        hourglass_time, mask_time, softmax_time, deformation_time, occlusion_time


class FirstOrderModel(KeypointBasedFaceModels):
    def __init__(self, config_path, for_onnx=False):
        super(FirstOrderModel, self).__init__()
        
        with open(config_path) as f:
            config = yaml.safe_load(f)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.for_onnx = for_onnx

        if self.for_onnx:
            # onnx generator
            self.generator = OcclusionAwareGenerator_ONNX(
                    **config['model_params']['generator_params'],
                    **config['model_params']['common_params'], for_onnx=True)
        else:
            # generator
            self.generator = OcclusionAwareGenerator(
                    **config['model_params']['generator_params'],
                    **config['model_params']['common_params'], for_onnx=False)

        if torch.cuda.is_available():
            self.generator.to(device)

        # keypoint detector
        self.kp_detector = KPDetector(
                **config['model_params']['kp_detector_params'],
                **config['model_params']['common_params'],
                for_onnx=for_onnx)
        if torch.cuda.is_available():
            self.kp_detector.to(device)

        # initialize weights
        checkpoint = config['checkpoint_params']['checkpoint_path']
        Logger.load_cpk(checkpoint, generator=self.generator, 
                kp_detector=self.kp_detector, device=device)

        # set to test mode
        self.generator.eval()
        self.kp_detector.eval()
        
        # placeholders for source information
        self.source_keypoints = None
        self.source = None


    def update_source(self, source_frame, source_keypoints):
        """ update the source and keypoints the frame is using 
            from the RGB source provided as input
        """
        transformed_source = np.array([img_as_float32(source_frame)])
        transformed_source = transformed_source.transpose((0, 3, 1, 2))
        self.source = torch.from_numpy(transformed_source)
        self.source_keypoints = source_keypoints 


    def extract_keypoints(self, frame):
        """ extract keypoints into a keypoint dictionary with/without jacobians
            from the provided RGB image 
        """
        transformed_frame = np.array([img_as_float32(frame)])
        transformed_frame = transformed_frame.transpose((0, 3, 1, 2))

        frame = torch.from_numpy(transformed_frame)
        if torch.cuda.is_available():
            frame = frame.cuda() 
        keypoint_struct = self.kp_detector(frame)
        if self.for_onnx:
            with torch.no_grad():
                keypoint_struct = {'value': keypoint_struct[0], 'jacobian': keypoint_struct[1]}

        # change to arrays and standardize
        # Note: keypoints are stored at key 'value' in FOM
        keypoint_struct['value'] = keypoint_struct['value'].data.cpu().numpy()[0]
        keypoint_struct['keypoints'] = keypoint_struct.pop('value')
        if 'jacobian' in keypoint_struct:
            try:
                keypoint_struct['jacobian'] = torch.int_repr(keypoint_struct['jacobian']).data.cpu().numpy()[0]
            except:
                keypoint_struct['jacobian'] = keypoint_struct['jacobian'].data.cpu().numpy()[0]
            keypoint_struct['jacobians'] = keypoint_struct.pop('jacobian')
        
        return keypoint_struct


    def convert_kp_dict_to_tensors(self, keypoint_dict):
        """ takes a keypoint dictionary and tensors the values appropriately """
        new_kp_dict = {}
        
        # Note: keypoints are stored at key 'value' in FOM
        new_kp_dict['value'] = torch.from_numpy(keypoint_dict['keypoints'])
        new_kp_dict['value'] = torch.unsqueeze(new_kp_dict['value'], 0)
        new_kp_dict['value'] = new_kp_dict['value'].float()

        if 'jacobians' in keypoint_dict:
            new_kp_dict['jacobian'] = torch.from_numpy(keypoint_dict['jacobians'])
            new_kp_dict['jacobian'] = torch.unsqueeze(new_kp_dict['jacobian'], 0)
            new_kp_dict['jacobian'] = new_kp_dict['jacobian'].float()
        
        if torch.cuda.is_available():
            for k in new_kp_dict.keys():
                new_kp_dict[k] = new_kp_dict[k].cuda() 

        return new_kp_dict


    def predict(self, target_keypoints):
        """ takes target keypoints and returns an RGB image for the prediction """
        assert(self.source_keypoints is not None)
        assert(self.source is not None)

        if torch.cuda.is_available():
            self.source = self.source.cuda()

        source_kp_tensors = self.convert_kp_dict_to_tensors(self.source_keypoints)
        target_kp_tensors = self.convert_kp_dict_to_tensors(target_keypoints)
        if self.for_onnx:
            with torch.no_grad():
                out = self.generator(self.source, \
                        kp_source_v=source_kp_tensors['value'], kp_driving_v=target_kp_tensors['value'],
                        kp_source_j=source_kp_tensors['jacobian'], kp_driving_j=target_kp_tensors['jacobian'],)
            prediction_cpu = out.data.cpu().numpy()
        else:
            out = self.generator(self.source, \
                    kp_source=source_kp_tensors, kp_driving=target_kp_tensors)
            prediction_cpu = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction_cpu, [0, 2, 3, 1])[0]
        return (255 * prediction).astype(np.uint8)


class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0, for_onnx=False):
        super(KPDetector, self).__init__()
        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.for_onnx = for_onnx
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """

        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        # grid = grid.to(f'cuda:{heatmap.get_device()}')
        value = (heatmap * grid).sum(dim=(2, 3))

        if self.for_onnx:
            kp = value
        else:
            kp = {'value': value}

        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        feature_map = self.quant(feature_map)
        prediction = self.kp(feature_map)
        prediction = self.dequant(prediction)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)
            heatmap = self.dequant(heatmap)
            jacobian_map = self.dequant(jacobian_map)
            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            jacobian = self.quant(jacobian)
            if not self.for_onnx:
                out['jacobian'] = jacobian

        if self.for_onnx:
            return out, jacobian
        else:
            return out


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
     {'block_expansion': 64, 'max_features': 1024, 'num_blocks': 5, 'scale_factor': 0.25}, True, False), enable_meausre=True):

    modules_to_fuse = get_coder_modules_to_fuse(len(model_fp32.dense_motion_network.hourglass.encoder.down_blocks), 
                                                prefix='dense_motion_network.hourglass.encoder.down_blocks')
    modules_to_fuse += get_coder_modules_to_fuse(len(model_fp32.dense_motion_network.hourglass.decoder.up_blocks),
                                                prefix='dense_motion_network.hourglass.decoder.up_blocks')    
    modules_to_fuse += [['first.conv', 'first.norm', 'first.relu']]
    modules_to_fuse += get_coder_modules_to_fuse(len(model_fp32.down_blocks), prefix='down_blocks')
    modules_to_fuse += get_coder_modules_to_fuse(len(model_fp32.up_blocks), prefix='up_blocks')
    modules_to_fuse += [['bottleneck.r0.conv1', 'bottleneck.r0.norm1'], ['bottleneck.r0.conv2', 'bottleneck.r0.norm2', 'bottleneck.r0.relu'],
                       ['bottleneck.r1.conv1', 'bottleneck.r1.norm1'], ['bottleneck.r1.conv2', 'bottleneck.r1.norm2', 'bottleneck.r1.relu'],
                       ['bottleneck.r2.conv1', 'bottleneck.r2.norm1'], ['bottleneck.r2.conv2', 'bottleneck.r2.norm2', 'bottleneck.r2.relu'],
                       ['bottleneck.r3.conv1', 'bottleneck.r3.norm1'], ['bottleneck.r3.conv2', 'bottleneck.r3.norm2', 'bottleneck.r3.relu'],
                       ['bottleneck.r4.conv1', 'bottleneck.r4.norm1'], ['bottleneck.r4.conv2', 'bottleneck.r4.norm2', 'bottleneck.r4.relu'],
                       ['bottleneck.r5.conv1', 'bottleneck.r5.norm1'], ['bottleneck.r5.conv2', 'bottleneck.r5.norm2', 'bottleneck.r5.relu']]
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
        bottleneck_times, up_blocks_1_times, up_blocks_2_times, final_times, sigmoid_times, tt = [], [], [], [], [], [], [], [], [], [], []

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

    print(f"using custom conv:{USE_FAST_CONV2}, using quantization:{USE_QUANTIZATION}, using float16:{USE_FLOAT_16}, resolution:{IMAGE_RESOLUTION}")
    print_average_and_std(first_times, "first_times")
    print_average_and_std(down_blocks_1_times, "down_blocks_1_times")
    print_average_and_std(down_blocks_2_times, "down_blocks_2_times")
    print_average_and_std(dense_morion_times, "dense_morion_times")
    print_average_and_std(deform_times, "deform_times")
    print_average_and_std(bottleneck_times, "bottleneck_times")
    print_average_and_std(up_blocks_1_times, "up_blocks_1_times")
    print_average_and_std(up_blocks_2_times, "up_blocks_2_times")
    print_average_and_std(final_times, "final_times")
    print_average_and_std(sigmoid_times, "sigmoid_times")
    print_average_and_std(tt, "total with print")


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
    
    print(f"using custom conv:{USE_FAST_CONV2}, using quantization:{USE_QUANTIZATION}, using float16:{USE_FLOAT_16}, resolution:{IMAGE_RESOLUTION}")
    print_average_and_std(heatmap_representation_times, "heatmap_representation_time")
    print_average_and_std(sparse_motion_times, "sparse_motion_time")
    print_average_and_std(create_deformed_source_times, "create_deformed_source_time")
    print_average_and_std(hourglass_times, "hourglass_time")
    print_average_and_std(mask_times, "mask_time")
    print_average_and_std(softmax_times, "softmax_time")
    print_average_and_std(deformation_times, "deformation_time")
    print_average_and_std(occlusion_times, "occlusion_time")
    print_average_and_std(tt, "total with print")


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
        quantize_enc()
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

