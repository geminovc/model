import torch
from torch import nn
import torch.nn.functional as F
from first_order_model.modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
import math

class SuperResolutionGenerator(nn.Module):
    """
    Generator that given low resolution target image, generates high resolution prediction
    """
    def __init__(self, num_channels, max_features, num_down_blocks,
                 num_bottleneck_blocks, upsample_factor=1, lr_size=64,
                 lr_features=32, generator_type='just_upsampler'):
        super(SuperResolutionGenerator, self).__init__()
        
        self.upsample_factor = upsample_factor
        upsample_levels = round(math.log(upsample_factor, 2))
        total_blocks = upsample_levels + num_down_blocks

        self.lr_first = SameBlock2d(num_channels, lr_features, kernel_size=(7, 7), padding=(3, 3))

  
        # regular decoder blocks with skip connections if need be
        up_blocks = []
        for i in range(total_blocks):
            in_features =  min(max_features, lr_features // (2 ** i))
            out_features = min(max_features, lr_features // (2 ** (i + 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = lr_features
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        final_input_features = out_features
        self.final = nn.Conv2d(final_input_features, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels


    def forward(self, driving_64x64):
        lr_encoded_features = self.lr_first(driving_64x64)
        
        out = self.bottleneck(lr_encoded_features)
        for i, block in enumerate(self.up_blocks):
            out = block(out)

        out = self.final(out)
        out = F.sigmoid(out)

        output_dict= {"prediction": out}

        return output_dict
