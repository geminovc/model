'''
This script computes the psnr of the data_dict values with keys in psnr_loss_apply_to. 

Peak signal-to-noise ratio (PSNR) is an engineering term for the ratio between the maximum possible
power of a signal and the power of corrupting noise that affects the fidelity of its representation.

Maximum possible power is measured with the maximum peak-to-peak ampliude of the input signals denoted by MAX_PP_AMP. 
The noise is computed with the root mean square of the differences of the input signals denoted by RMS_NOISE. 

psnr = 20 * log10 (MAX_PP_AMP/RMS_NOISE)

'''

# Third party
import torch
from torch import nn
import torch.nn.functional as F

# This project
from original_bilayer.runners import utils as rn_utils
import numpy as np
import math



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--psnr_loss_apply_to', type=str, help='what you want to apply psnr loss to')
    
    def __init__(self, args):
        super(LossWrapper, self).__init__()
        self.apply_to = [rn_utils.parse_str_to_list(s, sep=',') for s in rn_utils.parse_str_to_list(args.psnr_loss_apply_to, sep=';')]
    def PSNR(self, img1, img2):
        img1 = torch.mul(torch.add(img1, 1), 0.5).clamp(0, 1)
        img2 = torch.mul(torch.add(img2, 1), 0.5).clamp(0, 1)
        img1 = 255 * np.array(img1.cpu()) 
        img1 = img1.astype(np.uint8).astype(np.float32) # Typecasting to float values is really important
        img2 = 255 * np.array(img2.cpu()) 
        img2 = img2.astype(np.uint8).astype(np.float32)
        mse = np.mean((img1 - img2)**2)
        return torch.tensor(10 * math.log10( 255**2 /mse))
            

    def forward(self, data_dict, losses_dict):

        for i, (tensor_name, target_tensor_name) in enumerate(self.apply_to):
            real_imgs = data_dict[target_tensor_name]
            fake_imgs = data_dict[tensor_name]
            b, t = fake_imgs.shape[:2]
            fake_imgs = fake_imgs.view(b*t, *fake_imgs.shape[2:])

            if 'HalfTensor' in fake_imgs.type():  
                real_imgs = real_imgs.type(fake_imgs.type())

            real_imgs = real_imgs.view(b*t, *real_imgs.shape[2:])
            
            # Make these metrics and don't attach gradients to them
            losses_dict['G_PSNR'] = self.PSNR(fake_imgs.detach(), real_imgs.detach()).clone()
        return losses_dict
