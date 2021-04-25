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
        self.unet = UNet(19,3)

    def forward(
            self, 
            data_dict: dict,
            networks_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:
        # Information we need:        
        # Final images but with only the low frequency componenets
        # Done in inference generator because it makes more sense there. 
        if args.use_unet_only_hf:
            output =  self.unet(data_dict['warped_neural_textures'], data_dict) 
            data_dict['pred_target_imgs'] = output
            data_dict['pred_target_imgs_lf_detached'] = output
            data_dict['pred_target_delta_lf_rgbs'] = output
            
        if args.use_lf_with_unet:
            # Final images but with low frequency components detached. I'm not sure how to do this, so we will just use final images
            data_dict['pred_target_imgs'] = self.unet(data_dict['lf_hf_predictions'], data_dict) 
            # The actual images
            data_dict['pred_target_imgs_lf_detached'] = self.unet(data_dict['lf_detached_predictions'], data_dict)
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


class UNet(nn.Module):
    def __init__(self, input_channels=16, output_channels=3):
        super(UNet, self).__init__()
        self.down1 = down(input_channels, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 512)
        self.up1 = up(512, 512, output_pad=1, concat=False)
        self.up2 = up(1024, 512)
        self.up3 = up(768, 256)
        self.up4 = up(384, 128)
        self.up5 = up(192, output_channels, final=True)
    def forward(self, x, data_dict):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5, None)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        
        pred_target_imgs = x
        #rint(torch.mean(pred_target_imgs, dim=(0, 2, 3)))
        output_mean = torch.mean(pred_target_imgs, dim=(0, 2, 3))
        target_mean = torch.mean(data_dict['target_imgs'], dim=(0, 1, 3, 4))
        pred_target_imgs = pred_target_imgs - output_mean.view(1, 3, 1, 1) + target_mean.view(1, 3, 1, 1)
        return pred_target_imgs[None]

