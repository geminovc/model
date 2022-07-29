from torch import nn
import torch.nn.functional as F
import torch
from first_order_model.modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian
from first_order_model.modules.util import SameBlock2d 
import time
import numpy as np

class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, lr_features,
            num_channels, estimate_residual=False, num_pixel_features=0, estimate_occlusion_map=False, 
            scale_factor=1, kp_variance=0.01, run_at_256=False, concatenate_lr_frame_to_hourglass_input=False,
            concatenate_lr_frame_to_hourglass_output=False):
        super(DenseMotionNetwork, self).__init__()

        additional_features = lr_features if concatenate_lr_frame_to_hourglass_input else 0
        self.hourglass = Hourglass(block_expansion=block_expansion, 
                         in_features=(num_kp + 1) * (num_channels + 1 + num_pixel_features) + additional_features,
                         max_features=max_features, num_blocks=num_blocks)

        use_lr_frame = concatenate_lr_frame_to_hourglass_input or concatenate_lr_frame_to_hourglass_output
        additional_features = lr_features if concatenate_lr_frame_to_hourglass_output else 0
        self.mask = nn.Conv2d(self.hourglass.out_filters + additional_features, 
                              num_kp + 1, kernel_size=(7, 7), padding=(3, 3))

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters + additional_features, \
                    1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None
        
        if estimate_residual:
            self.residual = nn.Conv2d(self.hourglass.out_filters + additional_features, \
                    1, kernel_size=(7, 7), padding=(3, 3))
            self.num_pixel_features = num_pixel_features
        else:
            self.residual = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance
        self.run_at_256 = run_at_256
        self.concatenate_lr_frame_to_hourglass_input = concatenate_lr_frame_to_hourglass_input
        self.concatenate_lr_frame_to_hourglass_output = concatenate_lr_frame_to_hourglass_output

        if use_lr_frame:
            self.lr_first = SameBlock2d(num_channels, lr_features, kernel_size=(7, 7), padding=(3, 3))

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

        # saving state to reduce computation
        self.source_image = None
        self.update_source = True
        self.gaussian_source = None
        self.source_repeat = None
        self.identity_grid = None
    
    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[2:]
        pixel_heatmap = None
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        if self.update_source or self.gaussian_source == None:
            self.gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)

        heatmap = gaussian_driving - self.gaussian_source

        #adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], dtype=heatmap.dtype, device=heatmap.device)
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)

        # computing gaussians for pixel deltas, by multiplying each keypoint gaussian by 
        # associated pixel delta to smooth it out + background feature 0's
        if self.residual:
            pf_driving = kp_driving['pixel_features'].unsqueeze(3).unsqueeze(4)
            pf_source = kp_source['pixel_features'].unsqueeze(3).unsqueeze(4)

            gaussian_driving = gaussian_driving.unsqueeze(2)
            gaussian_source = self.gaussian_source.unsqueeze(2)

            pixel_gaussian_driving = pf_driving * gaussian_driving
            pixel_gaussian_source = pf_source * gaussian_source
            pixel_heatmap = pixel_gaussian_driving = pixel_gaussian_source
            
            zeros = torch.zeros(pixel_heatmap.shape[0], 1, pixel_heatmap.shape[2], 
                    spatial_size[0], spatial_size[1]).type(pixel_heatmap.type())
            pixel_heatmap = torch.cat([zeros, pixel_heatmap], dim=1)

        return heatmap, pixel_heatmap

    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        if self.identity_grid is None or self.update_source:
            self.identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())
            self.identity_grid = self.identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = self.identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)
        if 'jacobian' in kp_driving:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 2)

        #adding background feature
        identity_grid = self.identity_grid.repeat(bs, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        if self.source_repeat is None or self.update_source:
            self.source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
            self.source_repeat = self.source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        sparse_deformed = F.grid_sample(self.source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed

    def forward(self, source_image, kp_driving, kp_source, lr_frame = None):
        if self.run_at_256:
            source_image = F.interpolate(source_image, 256)

        if self.scale_factor != 1:
            source_image = self.down(source_image)

        if self.source_image is None:
            self.update_source = True
        else:
            self.update_source = not torch.all(self.source_image == source_image).item()

        if self.update_source:
            self.source_image = source_image

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation, pixel_representations = self.create_heatmap_representations(
                source_image, kp_driving, kp_source)
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        out_dict['sparse_deformed'] = deformed_source

        if self.residual:
            input = torch.cat([heatmap_representation, deformed_source, pixel_representations], dim=2)
        else:
            input = torch.cat([heatmap_representation, deformed_source], dim=2)

        input = input.view(bs, -1, h, w)
        if self.concatenate_lr_frame_to_hourglass_input:
            lr_frame_features = self.lr_first(lr_frame)
            input = torch.cat([input, lr_frame_features], dim = 1)

        prediction = self.hourglass(input)
        if self.concatenate_lr_frame_to_hourglass_output:
            lr_frame_features = self.lr_first(lr_frame)
            prediction = torch.cat([prediction, lr_frame_features], dim = 1)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict['deformation'] = deformation

        if self.residual:
            out_dict['residual'] = torch.sigmoid(self.residual(prediction))
        
        # Sec. 3.2 in the paper
        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
