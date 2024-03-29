"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import torch
from torch import nn
from torch.nn import functional as F
from first_order_model.modules.efficientnet_utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)

from first_order_model.modules.efficientnet_encoder import (
    MBConvBlock,
    VALID_MODELS
)


class EfficientNetDecoder(nn.Module):
    """EfficientNet model version of decoder, mimicked from the encoder
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:
        >>> import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None, lr_resolution=128, output_resolution=1024):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = []
        self._lr_feature_concat_idx = -1
        self._skip_connection_indices = []

        # reverse block order and input/output filters
        for block_args in reversed(blocks_args):
            input_filters = block_args.input_filters
            output_filters = block_args.output_filters

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(output_filters, self._global_params),
                output_filters=round_filters(input_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )
            self._blocks_args.append(block_args)


        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Reversing head and stem, so this is head
        image_size = (64, 64)
        in_channels = 256 # output of final block of encoder
        out_channels = round_filters(self._blocks_args[1].input_filters, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        
        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args[1:]:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of upsampling if the stride is > 1
            stride = block_args.stride
            stride = stride if isinstance(stride, int) else stride[0]
            if stride > 1:
                self._blocks.append(nn.Upsample(scale_factor=stride, mode='nearest'))
                image_size = calculate_output_image_size(image_size, block_args.stride, 'upsampling')
                block_args = block_args._replace(stride=1)
                if image_size[0] == lr_resolution:
                    self._lr_feature_concat_idx = len(self._blocks)
                    block_args = block_args._replace(input_filters=block_args.input_filters + 32)
                if image_size[0] > 128:
                    self._skip_connection_indices.append(len(self._blocks) + 1)
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            
            # adjust filter size for subsequent blocks
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))

        # stem because stem and head are reversed
        # self._upsample = nn.Upsample(scale_factor=2, mode='nearest')
        out_channels = 16 if output_resolution == 1024 else 32  # just before final layer which converts to rgb
        in_channels = round_filters(out_channels, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        
        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def upsample_features(self, inputs, lr_inputs, skip_connections):
        """use convolution layer to upsample features .

        Args:
            inputs (tensor): Input tensor.
            lr_inputs (tensor): Input from the LR encoder

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Head
        x = self._swish(self._bn1(self._conv_head(inputs)))

        # Blocks
        skip_idx = 0
        skip_reversed = skip_connections[::-1]
        for idx, block in enumerate(self._blocks):
            if idx == self._lr_feature_concat_idx:
                x = torch.cat([lr_inputs, x], dim=1)
            if idx in self._skip_connection_indices:
                if skip_idx < len(skip_connections):
                    skip = skip_reversed[skip_idx]
                    skip_idx += 1
                    x += skip
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            if isinstance(block, nn.Upsample):
                x = block(x)
            else:
                x = block(x, drop_connect_rate=drop_connect_rate)

        # Stem
        # x = self._upsample(x)
        x = self._swish(self._bn0(self._conv_stem(x)))

        return x

    def forward(self, inputs, lr_inputs, skip_connections={}):
        """EfficientNet's forward function.
           Calls upsample_features to upsample features, applies final linear layer, and returns features.

        Args:
            inputs (tensor): Input tensor.
            lr_inputs (tensor): Inputs from the LR encoder

        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x = self.upsample_features(inputs, lr_inputs, skip_connections)
        
        # Pooling and final linear layer
        """
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        """ 
        return x

    @classmethod
    def from_name(cls, model_name, lr_resolution=128, output_resolution=1024,
                  in_channels=3, **override_params):
        """Create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            lr_resolution (int): height and width for LR features
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        if output_resolution == 512:
            blocks_args = blocks_args[1:]
        model = cls(blocks_args, global_params, lr_resolution, output_resolution)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, lr_resolution=128, output_resolution=1024,
                        weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """Create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            lr_resolution (int): height and width of low res features
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(model_name, lr_resolution, output_resolution, num_classes=num_classes, **override_params)
        """
        load_pretrained_weights(model, model_name, weights_path=weights_path,
                                load_fc=(num_classes == 1000), advprop=advprop)
        """
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
