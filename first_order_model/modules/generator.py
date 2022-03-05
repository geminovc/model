import torch
from torch import nn
import torch.nn.functional as F
from first_order_model.modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from first_order_model.modules.dense_motion import DenseMotionNetwork
import math

class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, 
                 predict_pixel_features=False, num_pixel_features=0, 
                 run_at_256=False, upsample_factor=1, use_hr_skip_connections=False,
                 dense_motion_params=None, estimate_jacobian=False, encode_hr_input_with_additional_blocks=True):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels, 
                    estimate_residual=predict_pixel_features,
                    num_pixel_features=num_pixel_features,
                    estimate_occlusion_map=estimate_occlusion_map, 
                    **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.run_at_256 = run_at_256
        self.use_hr_skip_connections = use_hr_skip_connections
        assert (not run_at_256) or (run_at_256 and not use_hr_skip_connections and \
                not encode_hr_input_with_additional_blocks), "Cannot run generator at 256 and use HR input simultaneously"

        self.upsample_factor = upsample_factor
        upsample_levels = round(math.log(upsample_factor, 2))
        starting_depth = block_expansion // (2 ** upsample_levels)

        input_features = block_expansion if run_at_256 else starting_depth
        self.first = SameBlock2d(num_channels, input_features, kernel_size=(7, 7), padding=(3, 3))

        # enabling extra blocks for bringing higher resolution down
        hr_down_blocks = []
        if self.use_hr_skip_connections or encode_hr_input_with_additional_blocks:
            for i in range(upsample_levels):
                in_features = min(max_features, starting_depth * (2 ** i))
                out_features = min(max_features, starting_depth * (2 ** (i + 1)))
                hr_down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.hr_down_blocks = nn.ModuleList(hr_down_blocks)

        # regular encoder blcoks
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        # regular decoder blocks with skip connections if need be
        up_blocks = []
        offset = 2 if use_hr_skip_connections else 1
        for i in range(num_down_blocks):
            in_features = offset * min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        # add upsampling blocks at the end to increase resolution - will just be empty if there are no upsample levels
        hr_up_blocks = []
        for i in range(upsample_levels):
            in_features = offset * min(max_features, starting_depth * (2 ** (upsample_levels - i)))
            out_features = min(max_features, starting_depth * (2 ** (upsample_levels - i - 1)))
            hr_up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.hr_up_blocks = nn.ModuleList(hr_up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(starting_depth, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation), deformation

    def forward(self, source_image, kp_driving, kp_source):
        if self.run_at_256:
            resized_source_image = F.interpolate(source_image, 256)
        else:
            resized_source_image = source_image
        
        # Encoding (downsampling) part
        out = self.first(resized_source_image)
        skip_connections = [out] if self.use_hr_skip_connections else []
        for block in self.hr_down_blocks: 
            out = block(out)
            if self.use_hr_skip_connections:
                skip_connections.append(out)
        
        for block in self.down_blocks: 
            out = block(out)
            if self.use_hr_skip_connections:
                skip_connections.append(out)

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
                out = out * occlusion_map

            if 'residual' in dense_motion:
                out += dense_motion['residual']

            output_dict["deformed"], deformation = self.deform_input(source_image, deformation)
            output_dict["deformation"] = deformation

        # Decoding part
        out = self.bottleneck(out)
        for block in self.up_blocks:
            if self.use_hr_skip_connections:
                skip = skip_connections.pop()
                skip, _ = self.deform_input(skip, deformation)
                out = torch.cat([out, skip], dim=1)
            out = block(out)
        
        for block in self.hr_up_blocks:
            if self.use_hr_skip_connections:
                skip = skip_connections.pop()
                skip, _ = self.deform_input(skip, deformation)
                out = torch.cat([out, skip], dim=1)
            out = block(out)

        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict
