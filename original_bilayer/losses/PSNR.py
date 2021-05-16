# Third party
import torch
from torch import nn
import torch.nn.functional as F

# This project
from runners import utils as rn_utils



class LossWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--psnr_loss_apply_to', type=str, help='what you want to apply psnr loss to')
    
    def __init__(self, args):
        super(LossWrapper, self).__init__()
        self.apply_to = [rn_utils.parse_str_to_list(s, sep=',') for s in rn_utils.parse_str_to_list(args.psnr_loss_apply_to, sep=';')]
    def PSNR(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(2.0 / torch.sqrt(mse))
            

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
