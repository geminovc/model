# Copied from https://github.com/SSRSGJYD/NeuralTexture/blob/master/model/unet.py to make sure I get the details right
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# This project
from runners import utils as rn_utils
from networks import utils as nt_utils



class NetworkWrapper(nn.Module):


    def __init__(self, args):
        super(NetworkWrapper, self).__init__()
        # Initialize options
        self.unet = UNet()

    def forward(
            self, 
            data_dict: dict,
            networks_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:
        data_dict = self.unet(data_dict)
        return data_dict


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, output_pad=0, concat=True, final=False):
        super(up, self).__init__()
        self.concat = concat
        self.final = final
        if self.final:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_pad),
                nn.InstanceNorm2d(out_ch),
                nn.Tanh()
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, output_padding=output_pad),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x1, x2):
        if self.concat:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x1 = torch.cat((x2, x1), dim=1)
        x1 = self.conv(x1)
        return x1


class oneConv(nn.Module):
    def __init__(self, input_channels=16, output_channels=3):
        super(UNet, self).__init__()
        self.simpleconv = nn.Conv2d(input_channels, output_channels, 1)
    def forward(self, data_dict):
        x = data_dict['warped_neural_textures']
        x = self.simpleconv(x)
        pred_target_lf_rgbs = data_dict['warped_neural_textures'][:,:3]
        pred_target_imgs = x
        #rint(torch.mean(pred_target_imgs, dim=(0, 2, 3)))
        output_mean = torch.mean(pred_target_imgs, dim=(0, 2, 3))
        target_mean = torch.mean(data_dict['target_imgs'], dim=(0, 1, 3, 4))
        pred_target_imgs = pred_target_imgs - output_mean.view(1, 3, 1, 1) + target_mean.view(1, 3, 1, 1)
        
        # Information we need:        
        # Final images but with only the low frequency componenets
        data_dict['pred_target_delta_lf_rgbs'] = pred_target_imgs[None]
        
        # Final images but with low frequency components detached. I'm not sure how to do this, so we will just use final images
        data_dict['pred_target_imgs_lf_detached'] = pred_target_imgs[None]
        
        # The actual images
        data_dict['pred_target_imgs'] = pred_target_imgs[None]
        return data_dict

