import sys
sys.path.append("..")
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
from frames_dataset import DatasetRepeater
from tqdm import trange
from sync_batchnorm import DataParallelWithCallback
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
from modules.discriminator import MultiScaleDiscriminator
from frames_dataset import FramesDataset
from torchvision import models
import numpy as np
from torch.autograd import grad


QUANT_ENGINE = 'fbgemm'
USE_FAST_CONV2 = False
USE_FLOAT_16 = False
USE_QUANTIZATION = False
IMAGE_RESOLUTION = 256
USE_CUDA = False
NUM_RUNS = 1000

_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])
_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])
_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


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


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        kp_driving = generated['kp_driving']
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, 'Previous result has\'t been fetched.'
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()

            res = self._result
            self._result = None
            return res


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i*2:i*2+2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    r"""Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm1d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 5d input that is seen as a mini-batch
    of 4d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm3d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
    or Spatio-temporal BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm3d, self)._check_input_dim(input)


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


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


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


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None,
                 optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None, 
                 device='gpu'):
        if device == torch.device('cpu'):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)

        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])

        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def draw_deformation_heatmap(self, deformation):
        h, w = 256, 256
        deformation_heatmap = np.zeros((1, h, w, 3))
        for x in range(h):
            for y in range(w):
                input_location = deformation[0][x][y] 
                deformation_heatmap[0][x][y][0] = (input_location[0] + 1.0) / 2.0
                deformation_heatmap[0][x][y][1] = (input_location[1] + 1.0) / 2.0
        return deformation_heatmap



    def visualize(self, driving, source, out):
        images = []

        # Source image with keypoints
        source = source.data.cpu()
        kp_source = out['kp_source']['value'].data.cpu().numpy()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # Equivariance visualization
        if 'transformed_frame' in out:
            transformed = out['transformed_frame'].data.cpu().numpy()
            transformed = np.transpose(transformed, [0, 2, 3, 1])
            transformed_kp = out['transformed_kp']['value'].data.cpu().numpy()
            images.append((transformed, transformed_kp))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['value'].data.cpu().numpy()
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        # Deformed image
        if 'deformed' in out:
            deformed = out['deformed'].data.cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)

        # deformation heatmap
        if 'deformation' in out:
            deformation = out['deformation'].data.cpu().numpy()
            heatmap = self.draw_deformation_heatmap(deformation)
            images.append(heatmap)

        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['value'].data.cpu().numpy()
            images.append((prediction, kp_norm))
        images.append(prediction)


        ## Occlusion map
        if 'occlusion_map' in out:
            occlusion_map = out['occlusion_map'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)

        # Deformed images according to each individual transform
        if 'sparse_deformed' in out:
            full_mask = []
            for i in range(out['sparse_deformed'].shape[1]):
                image = out['sparse_deformed'][:, i].data.cpu()
                image = F.interpolate(image, size=source.shape[1:3])
                mask = out['mask'][:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)
                mask = F.interpolate(mask, size=source.shape[1:3])
                image = np.transpose(image.numpy(), (0, 2, 3, 1))
                mask = np.transpose(mask.numpy(), (0, 2, 3, 1))

                if i != 0:
                    color = np.array(self.colormap((i - 1) / (out['sparse_deformed'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 1, 3))

                images.append(image)
                if i != 0:
                    images.append(mask * color)
                else:
                    images.append(mask)

                full_mask.append(mask * color)

            images.append(sum(full_mask))

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


def kp2gaussian(kp_value, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp_value

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)
    # coordinate_grid = coordinate_grid.to(f'cuda:{mean.get_device()}')
    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size


def get_params(model):
    params = []
    for name, mod in model.named_modules():                             
        if isinstance(mod, torch.nn.quantized.Conv2d):                              
            weight, bias = mod._weight_bias()
            params.append(weight)
            params.append(bias)
    return params  


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


def quantize_model(model_fp32, modules_to_fuse, x0, x1=None, x2=None, x3=None, x4=None, enable_meausre=False):
    model_fp32.eval()
    model_fp32.qconfig = torch.quantization.get_default_qconfig(QUANT_ENGINE)
    model_fp32_fused = torch.quantization.fuse_modules(model_fp32, modules_to_fuse)
    model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
    model_int8 = torch.quantization.convert(model_fp32_prepared)

    if enable_meausre:
        print_model_info(model_fp32, "model_fp32", x0, x1, x2, x3, x4)
        print_model_info(model_int8, "model_int8", x0, x1, x2, x3, x4)

    return model_int8


def print_average_and_std(test_list, name):
    mean = sum(test_list) / len(test_list)
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list)
    res = variance ** 0.5
    print(f"{name}:: mean={round(mean, 6)}, std={round(res / mean * 100, 6)}%")


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


def print_model_info(model, model_name, x0, x1=None, x2=None, x3=None, x4=None):
        print_size_of_model(model, label=model_name)
        tt = []
        for i in range(0, NUM_RUNS):
            print(f"run #{i}")
            if x1 != None:
                start_time = time.time()
                res = model(x0, {'value':x1, 'jacobian':x2}, {'value':x3, 'jacobian':x4})
                tt.append(time.time() - start_time)
            else:
                start_time = time.time()
                res = model(x0)
                tt.append(time.time() - start_time)
        print_average_and_std(tt, f"Average inference time on {model_name}")


def quantize_generator(model_fp32=OcclusionAwareGenerator(3, 10, 64, 512, 2, 6, True,
     {'block_expansion': 64, 'max_features': 1024, 'num_blocks': 5, 'scale_factor': 0.25}, True, False), enable_meausre=True):

    modules_to_fuse = [['dense_motion_network.hourglass.encoder.down_blocks.0.conv', 'dense_motion_network.hourglass.encoder.down_blocks.0.norm', 'dense_motion_network.hourglass.encoder.down_blocks.0.relu'],
                       ['dense_motion_network.hourglass.encoder.down_blocks.1.conv', 'dense_motion_network.hourglass.encoder.down_blocks.1.norm', 'dense_motion_network.hourglass.encoder.down_blocks.1.relu'],
                       ['dense_motion_network.hourglass.encoder.down_blocks.2.conv', 'dense_motion_network.hourglass.encoder.down_blocks.2.norm', 'dense_motion_network.hourglass.encoder.down_blocks.2.relu'],
                       ['dense_motion_network.hourglass.encoder.down_blocks.3.conv', 'dense_motion_network.hourglass.encoder.down_blocks.3.norm', 'dense_motion_network.hourglass.encoder.down_blocks.3.relu'],
                       ['dense_motion_network.hourglass.encoder.down_blocks.4.conv', 'dense_motion_network.hourglass.encoder.down_blocks.4.norm', 'dense_motion_network.hourglass.encoder.down_blocks.4.relu'],
                       ['dense_motion_network.hourglass.decoder.up_blocks.0.conv', 'dense_motion_network.hourglass.decoder.up_blocks.0.norm', 'dense_motion_network.hourglass.decoder.up_blocks.0.relu'],
                       ['dense_motion_network.hourglass.decoder.up_blocks.1.conv', 'dense_motion_network.hourglass.decoder.up_blocks.1.norm', 'dense_motion_network.hourglass.decoder.up_blocks.1.relu'],
                       ['dense_motion_network.hourglass.decoder.up_blocks.2.conv', 'dense_motion_network.hourglass.decoder.up_blocks.2.norm', 'dense_motion_network.hourglass.decoder.up_blocks.2.relu'],
                       ['dense_motion_network.hourglass.decoder.up_blocks.3.conv', 'dense_motion_network.hourglass.decoder.up_blocks.3.norm', 'dense_motion_network.hourglass.decoder.up_blocks.3.relu'],
                       ['dense_motion_network.hourglass.decoder.up_blocks.4.conv', 'dense_motion_network.hourglass.decoder.up_blocks.4.norm', 'dense_motion_network.hourglass.decoder.up_blocks.4.relu'],
                       ['first.conv', 'first.norm', 'first.relu'],
                       ['down_blocks.0.conv', 'down_blocks.0.norm', 'down_blocks.0.relu'],
                       ['down_blocks.1.conv', 'down_blocks.1.norm', 'down_blocks.1.relu'],
                       ['up_blocks.0.conv', 'up_blocks.0.norm', 'up_blocks.0.relu'],
                       ['up_blocks.1.conv', 'up_blocks.1.norm', 'up_blocks.1.relu'],
                       ['bottleneck.r0.conv1', 'bottleneck.r0.norm1'], ['bottleneck.r0.conv2', 'bottleneck.r0.norm2', 'bottleneck.r0.relu'],
                       ['bottleneck.r1.conv1', 'bottleneck.r1.norm1'], ['bottleneck.r1.conv2', 'bottleneck.r1.norm2', 'bottleneck.r1.relu'],
                       ['bottleneck.r2.conv1', 'bottleneck.r2.norm1'], ['bottleneck.r2.conv2', 'bottleneck.r2.norm2', 'bottleneck.r2.relu'],
                       ['bottleneck.r3.conv1', 'bottleneck.r3.norm1'], ['bottleneck.r3.conv2', 'bottleneck.r3.norm2', 'bottleneck.r3.relu'],
                       ['bottleneck.r4.conv1', 'bottleneck.r4.norm1'], ['bottleneck.r4.conv2', 'bottleneck.r4.norm2', 'bottleneck.r4.relu'],
                       ['bottleneck.r5.conv1', 'bottleneck.r5.norm1'], ['bottleneck.r5.conv2', 'bottleneck.r5.norm2', 'bottleneck.r5.relu']]
    
    x0, x1, x2, x3, x4 = get_random_inputs("generator")
    model_int8 = quantize_model(model_fp32, modules_to_fuse, x0, x1, x2, x3, x4, enable_meausre)

    return model_int8


def quantize_kp_detector(model_fp32=KPDetector(32, 10, 3, 1024, 5, 0.1, True, 0.25, False, 0, False), enable_meausre=True):
    modules_to_fuse = [['predictor.encoder.down_blocks.0.conv', 'predictor.encoder.down_blocks.0.norm', 'predictor.encoder.down_blocks.0.relu'],
                       ['predictor.encoder.down_blocks.1.conv', 'predictor.encoder.down_blocks.1.norm', 'predictor.encoder.down_blocks.1.relu'],
                       ['predictor.encoder.down_blocks.2.conv', 'predictor.encoder.down_blocks.2.norm', 'predictor.encoder.down_blocks.2.relu'],
                       ['predictor.encoder.down_blocks.3.conv', 'predictor.encoder.down_blocks.3.norm', 'predictor.encoder.down_blocks.3.relu'],
                       ['predictor.encoder.down_blocks.4.conv', 'predictor.encoder.down_blocks.4.norm', 'predictor.encoder.down_blocks.4.relu'],
                       ['predictor.decoder.up_blocks.0.conv', 'predictor.decoder.up_blocks.0.norm', 'predictor.decoder.up_blocks.0.relu'],
                       ['predictor.decoder.up_blocks.1.conv', 'predictor.decoder.up_blocks.1.norm', 'predictor.decoder.up_blocks.1.relu'],
                       ['predictor.decoder.up_blocks.2.conv', 'predictor.decoder.up_blocks.2.norm', 'predictor.decoder.up_blocks.2.relu'],
                       ['predictor.decoder.up_blocks.3.conv', 'predictor.decoder.up_blocks.3.norm', 'predictor.decoder.up_blocks.3.relu'],
                       ['predictor.decoder.up_blocks.4.conv', 'predictor.decoder.up_blocks.4.norm', 'predictor.decoder.up_blocks.4.relu'],]

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
    modules_to_fuse = [['down_blocks.0.conv', 'down_blocks.0.norm', 'down_blocks.0.relu'],
                       ['down_blocks.1.conv', 'down_blocks.1.norm', 'down_blocks.1.relu'],
                       ['down_blocks.2.conv', 'down_blocks.2.norm', 'down_blocks.2.relu'],
                       ['down_blocks.3.conv', 'down_blocks.3.norm', 'down_blocks.3.relu'],
                       ['down_blocks.4.conv', 'down_blocks.4.norm', 'down_blocks.4.relu']]
    
    model_int8 = quantize_model(model_fp32, modules_to_fuse, torch.randn(1, 44, 64, 64), enable_meausre=enable_meausre)
    return model_int8


def quantize_dec(model_fp32=Decoder(64, 44, 5, 1024), enable_meausre=True):
    modules_to_fuse = [['up_blocks.0.conv', 'up_blocks.0.norm', 'up_blocks.0.relu'],
                       ['up_blocks.1.conv', 'up_blocks.1.norm', 'up_blocks.1.relu'],
                       ['up_blocks.2.conv', 'up_blocks.2.norm', 'up_blocks.2.relu'],
                       ['up_blocks.3.conv', 'up_blocks.3.norm', 'up_blocks.3.relu'],
                       ['up_blocks.4.conv', 'up_blocks.4.norm', 'up_blocks.4.relu']]

    x0 = [torch.randn(1, 44, 64, 64), torch.randn(1, 128, 32, 32),
                  torch.randn(1, 256, 16, 16), torch.randn(1, 512, 8, 8),
                  torch.randn(1, 1024, 4, 4), torch.randn(1, 1024, 2, 2)]
    model_int8 = quantize_model(model_fp32, modules_to_fuse, x0, enable_meausre=enable_meausre)
    return model_int8


def quantize_hrglass(model_fp32=Hourglass(64, 44, 5, 1024), enable_meausre=True):
    modules_to_fuse = [['encoder.down_blocks.0.conv', 'encoder.down_blocks.0.norm', 'encoder.down_blocks.0.relu'],
                       ['encoder.down_blocks.1.conv', 'encoder.down_blocks.1.norm', 'encoder.down_blocks.1.relu'],
                       ['encoder.down_blocks.2.conv', 'encoder.down_blocks.2.norm', 'encoder.down_blocks.2.relu'],
                       ['encoder.down_blocks.3.conv', 'encoder.down_blocks.3.norm', 'encoder.down_blocks.3.relu'],
                       ['encoder.down_blocks.4.conv', 'encoder.down_blocks.4.norm', 'encoder.down_blocks.4.relu'],
                       ['decoder.up_blocks.0.conv', 'decoder.up_blocks.0.norm', 'decoder.up_blocks.0.relu'],
                       ['decoder.up_blocks.1.conv', 'decoder.up_blocks.1.norm', 'decoder.up_blocks.1.relu'],
                       ['decoder.up_blocks.2.conv', 'decoder.up_blocks.2.norm', 'decoder.up_blocks.2.relu'],
                       ['decoder.up_blocks.3.conv', 'decoder.up_blocks.3.norm', 'decoder.up_blocks.3.relu'],
                       ['decoder.up_blocks.4.conv', 'decoder.up_blocks.4.norm', 'decoder.up_blocks.4.relu']]

    model_int8 = quantize_model(model_fp32, modules_to_fuse, torch.randn(1, 44, 64, 64), enable_meausre=enable_meausre)
    return model_int8


def quantize_dense_motion(model_fp32=DenseMotionNetwork(64, 5, 1024, 10, 3, True, 0.25, 0.01, False), enable_meausre=True):
    modules_to_fuse = [['hourglass.encoder.down_blocks.0.conv', 'hourglass.encoder.down_blocks.0.norm', 'hourglass.encoder.down_blocks.0.relu'],
                       ['hourglass.encoder.down_blocks.1.conv', 'hourglass.encoder.down_blocks.1.norm', 'hourglass.encoder.down_blocks.1.relu'],
                       ['hourglass.encoder.down_blocks.2.conv', 'hourglass.encoder.down_blocks.2.norm', 'hourglass.encoder.down_blocks.2.relu'],
                       ['hourglass.encoder.down_blocks.3.conv', 'hourglass.encoder.down_blocks.3.norm', 'hourglass.encoder.down_blocks.3.relu'],
                       ['hourglass.encoder.down_blocks.4.conv', 'hourglass.encoder.down_blocks.4.norm', 'hourglass.encoder.down_blocks.4.relu'],
                       ['hourglass.decoder.up_blocks.0.conv', 'hourglass.decoder.up_blocks.0.norm', 'hourglass.decoder.up_blocks.0.relu'],
                       ['hourglass.decoder.up_blocks.1.conv', 'hourglass.decoder.up_blocks.1.norm', 'hourglass.decoder.up_blocks.1.relu'],
                       ['hourglass.decoder.up_blocks.2.conv', 'hourglass.decoder.up_blocks.2.norm', 'hourglass.decoder.up_blocks.2.relu'],
                       ['hourglass.decoder.up_blocks.3.conv', 'hourglass.decoder.up_blocks.3.norm', 'hourglass.decoder.up_blocks.3.relu'],
                       ['hourglass.decoder.up_blocks.4.conv', 'hourglass.decoder.up_blocks.4.norm', 'hourglass.decoder.up_blocks.4.relu']]


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

