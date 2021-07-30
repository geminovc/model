# Copied from https://github.com/SSRSGJYD/NeuralTexture/blob/master/model/unet.py to make sure I get the details right
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# This project
from runners import utils as rn_utils
from networks import utils as nt_utils



class NetworkWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--unet_input_channels',  default = 16, type=int, help='unet input channels')
        parser.add('--unet_output_channels', default = 3,  type=int, help='unet output channels')
        args, _ = parser.parse_known_args()
    def __init__(self, args):
        super(NetworkWrapper, self).__init__()
        self.args = args
        # Initialize options
        self.unet = UNet(args.unet_input_channels,args.unet_output_channels)

    def forward(
            self, 
            data_dict: dict,
            networks_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:
        pred_target_imgs =  self.unet(data_dict['unet_input'], data_dict) 
        pred_target_delta_lf_rgbs_detached = self.unet(data_dict['lf_detached_inputs'], data_dict) 
        target_pose_embeds = data_dict['target_pose_embeds']

        # Needed to reshape the outputs
        b, t = target_pose_embeds.shape[:2]
        reshape_target_data = lambda data: data.view(b, t, *data.shape[1:])

        # If we are going to apply masks to focus on the foreground or the face
        if self.args.inf_apply_masks and self.args.inf_pred_segmentation:
            # Predicted segmentation masks are applied to target images
            target_imgs = data_dict['target_imgs']
            pred_target_segs = data_dict['pred_target_segs']
            pred_target_masks = pred_target_segs.detach()

            # Apply masks to target imgs
            target_imgs = target_imgs * pred_target_masks + (-1) * (1 - pred_target_masks)

            # Apply masks to predicted version of target images
            pred_target_imgs = pred_target_imgs * pred_target_masks + (-1) * (1 - pred_target_masks)

            # Apply masks to the hf-only version of target imgs
            pred_target_delta_lf_rgbs_detached = pred_target_delta_lf_rgbs_detached * pred_target_masks + (-1) * (1 - pred_target_masks)

            # Save the results (no reshaping needed because target images are already reshaped)
            data_dict['target_imgs'] = target_imgs

        # Shift the mean of the output images to match the target images
        # This was needed because the outputs were always dimmer than the target and this seems
        # to be the easiest fix
        output_mean = torch.mean(pred_target_imgs, dim=(0, 1, 3, 4))
        target_mean = torch.mean(target_imgs, dim=(0, 1, 3, 4))
        pred_target_imgs = pred_target_imgs - output_mean.view(1, 1, 3, 1, 1) + target_mean.view(1, 1,3, 1, 1)      
        output_mean_d = torch.mean(pred_target_delta_lf_rgbs_detached, dim=(0,1, 3, 4))
        pred_target_delta_lf_rgbs_detached = pred_target_delta_lf_rgbs_detached - output_mean_d.view(1, 1, 3, 1, 1) + target_mean.view(1, 1, 3, 1, 1)      
        
        # Save the resulting images
        data_dict['pred_target_imgs'] = pred_target_imgs
        data_dict['unet_input'] = reshape_target_data(data_dict['unet_input'])
        data_dict['pred_target_imgs_lf_detached'] = pred_target_delta_lf_rgbs_detached

        return data_dict
    
    @torch.no_grad()
    def visualize_outputs(self, data_dict):
        # All visualization is done in the inference generator
        visuals = []
        visuals += [data_dict['pred_target_imgs']]
        return visuals

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
        target_pose_embeds=data_dict['target_pose_embeds'] 
        b, t = target_pose_embeds.shape[:2]
        reshape_target_data = lambda data: data.view(b, t, *data.shape[1:])
        pred_target_imgs = x
        #rint(torch.mean(pred_target_imgs, dim=(0, 2, 3)))
        return reshape_target_data(pred_target_imgs)


