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
        self.oneConv = oneConv(args)

    def forward(
            self, 
            data_dict: dict,
            networks_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:
        output =  self.unet(data_dict['unet_input'], data_dict) 
        data_dict['pred_target_imgs'] = output
        output_lf_detached =  self.unet(data_dict['lf_detached_inputs'], data_dict) 
        data_dict['pred_target_imgs_lf_detached'] = output_lf_detached
        return data_dict

class oneConv(nn.Module):
    def __init__(self, input_channels=16, output_channels=3):
        super(UNet, self).__init__()
        self.simpleconv = nn.Conv2d(input_channels, output_channels, 1)
    def forward(self, warped_neural_textures, data_dict):
        x = self.simpleconv(warped_neural_textures)
        target_pose_embeds=data_dict['target_pose_embeds'] 
        b, t = target_pose_embeds.shape[:2]
        reshape_target_data = lambda data: data.view(b, t, *data.shape[1:])
        pred_target_imgs = x
        #rint(torch.mean(pred_target_imgs, dim=(0, 2, 3)))
        output_mean = torch.mean(pred_target_imgs, dim=(0, 2, 3))
        target_mean = torch.mean(data_dict['target_imgs'], dim=(0, 1, 3, 4))
        pred_target_imgs = pred_target_imgs - output_mean.view(1, 3, 1, 1) + target_mean.view(1, 3, 1, 1)
        return reshape_target_data(pred_target_imgs)

