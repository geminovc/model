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
                 dense_motion_params=None, estimate_jacobian=False, encode_hr_input_with_additional_blocks=False,
                 use_lr_video=False, lr_features=32, lr_size=64,
                 hr_features=16, generator_type='occlusion_aware'):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                    lr_features=lr_features,
                    estimate_residual=predict_pixel_features,
                    num_pixel_features=num_pixel_features,
                    estimate_occlusion_map=estimate_occlusion_map, 
                    **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.run_at_256 = run_at_256
        self.use_hr_skip_connections = use_hr_skip_connections
        self.encode_hr_input_with_additional_blocks = encode_hr_input_with_additional_blocks
        self.generator_type = generator_type
        self.lr_size = lr_size
        
        if dense_motion_params.get('concatenate_lr_frame_to_hourglass', False) \
                or dense_motion_params.get('concatenate_lr_frame_to_prediction', False):
            self.use_lr_video = False 
        else:
            self.use_lr_video = use_lr_video


        if use_hr_skip_connections:
            assert run_at_256, "Skip connections require parallel 256 FOM pipeline"
        else:
            assert (not run_at_256) or (run_at_256 and not encode_hr_input_with_additional_blocks), \
                "Cannot run downsample blocks and running at 256 fom simultaneously"

        assert (not run_at_256) or (run_at_256 and upsample_factor > 1), \
                "Need to upsample appropriately if generator runs at 256"
        
        self.upsample_factor = upsample_factor
        upsample_levels = round(math.log(upsample_factor, 2))
        hr_starting_depth = hr_features if use_hr_skip_connections else block_expansion // (2 ** upsample_levels) 

        # first layer either designed for 256x256 input or HR input
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        if self.use_hr_skip_connections or self.encode_hr_input_with_additional_blocks:
            input_features = hr_features if self.use_hr_skip_connections else hr_starting_depth
            self.hr_first = SameBlock2d(num_channels, input_features, kernel_size=(7, 7), padding=(3, 3))

        # first layer for LR input
        if self.use_lr_video:
            self.lr_first = SameBlock2d(num_channels, lr_features, kernel_size=(7, 7), padding=(3, 3))

        # enabling extra blocks for bringing higher resolution down
        hr_down_blocks = []
        if self.use_hr_skip_connections or self.encode_hr_input_with_additional_blocks:
            for i in range(upsample_levels):
                in_features = min(max_features, hr_starting_depth * (2 ** i))
                out_features = min(max_features, hr_starting_depth * (2 ** (i + 1)))
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
        for i in range(num_down_blocks):
            in_features =  min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            if i == math.log(self.lr_size / 64, 2) \
                    and self.use_lr_video and self.generator_type == 'occlusion_aware':
                in_features += lr_features

            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        # add upsampling blocks at the end to increase resolution - will just be empty if there are no upsample levels
        hr_up_blocks = []
        offset = 2 if use_hr_skip_connections else 1
        for i in range(upsample_levels):
            in_features = offset * min(max_features, hr_starting_depth * (2 ** (upsample_levels - i)))
            if i == (math.log(self.lr_size / 64, 2) - len(self.up_blocks)) \
                    and self.use_lr_video and self.generator_type == 'occlusion_aware':
                in_features += lr_features
            out_features = min(max_features, hr_starting_depth * (2 ** (upsample_levels - i - 1)))
            hr_up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.hr_up_blocks = nn.ModuleList(hr_up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
        final_input_features = hr_starting_depth if upsample_levels > 0 else block_expansion
        self.final = nn.Conv2d(final_input_features, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map

        # upsampling blocks for LF superresolution if using it
        if generator_type == "split_hf_lf":
            sr_up_blocks = []
            total_blocks = upsample_levels + num_down_blocks
            for i in range(total_blocks):
                in_features =  min(max_features, lr_features // (2 ** i))
                out_features = min(max_features, lr_features // (2 ** (i + 1)))
                sr_up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.sr_up_blocks = nn.ModuleList(sr_up_blocks)

            self.sr_bottleneck = torch.nn.Sequential()
            in_features = lr_features
            for i in range(num_bottleneck_blocks):
                self.sr_bottleneck.add_module('r' + str(i), \
                        ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

            sr_final_input_features = out_features
            self.sr_final = nn.Conv2d(sr_final_input_features, num_channels, kernel_size=(7, 7), padding=(3, 3))

        self.num_channels = num_channels
        self.source_image = None
        self.update_source = True
        self.encoder_output = None
        self.skip_connections = None

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation), deformation

    def forward(self, source_image, kp_driving, kp_source, update_source=False, driving_lr=None):
        if self.source_image is None:
            self.update_source = True
        else:
            self.update_source = update_source

        if self.update_source:
            self.source_image = source_image
        
        # Encoding (downsampling) part
        if self.encoder_output is None or self.update_source:
            if self.run_at_256:
                resized_source_image = F.interpolate(source_image, 256)
            else:
                resized_source_image = source_image
            
            hr_out = None
            if self.use_hr_skip_connections or self.encode_hr_input_with_additional_blocks:
                hr_out = self.hr_first(source_image)
                self.skip_connections = [hr_out] if self.use_hr_skip_connections else []
                for block in self.hr_down_blocks:
                    hr_out = block(hr_out)
                    if self.use_hr_skip_connections:
                        self.skip_connections.append(hr_out)
            
            out = hr_out if self.encode_hr_input_with_additional_blocks else self.first(resized_source_image)
            for block in self.down_blocks: 
                out = block(out)

            self.encoder_output = out

        # lr target image encoding
        if self.use_lr_video:
            lr_encoded_features = self.lr_first(driving_lr)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source, lr_frame=driving_lr)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out, _ = self.deform_input(self.encoder_output, deformation)

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
        for i, block in enumerate(self.up_blocks):
            if i == math.log(self.lr_size / 64, 2) \
                    and self.use_lr_video and self.generator_type == "occlusion_aware":
                out_shape = list(out.shape)
                out_shape.pop(1)
                lr_shape = list(lr_encoded_features.shape)
                lr_shape.pop(1)
                assert out_shape == lr_shape, "Dimensions mismatch in LR video input and rest of pipeline"
                out = torch.cat([out, lr_encoded_features], dim=1)
            out = block(out)

        for i in range(len(self.hr_up_blocks)):
            block = self.hr_up_blocks[i]
            if i == (math.log(self.lr_size / 64, 2) - len(self.up_blocks)) \
                    and self.use_lr_video and self.generator_type == "occlusion_aware":
                out_shape = list(out.shape)
                out_shape.pop(1)
                lr_shape = list(lr_encoded_features.shape)
                lr_shape.pop(1)
                assert out_shape == lr_shape, "Dimensions mismatch in LR video input and rest of pipeline"
                out = torch.cat([out, lr_encoded_features], dim=1)
            if self.use_hr_skip_connections:
                skip = self.skip_connections[len(self.skip_connections) - 1 - i]
                skip, _ = self.deform_input(skip, deformation)
                out = torch.cat([out, skip], dim=1)
            out = block(out)

        out = self.final(out)
        out = F.sigmoid(out)

        # use LF SR pipeline if required and add it to above pipeline result
        if self.generator_type == "split_hf_lf":
            lf_out = self.sr_bottleneck(lr_encoded_features)
            for i, block in enumerate(self.sr_up_blocks):
                lf_out = block(lf_out)

            lf_out = self.sr_final(lf_out)
            lf_out = F.sigmoid(lf_out)
            output_dict["prediction_lf"] = lf_out
            output_dict["prediction_hf"] = out
            output_dict["prediction_lf_detached"] = lf_out.detach() + out
            out = out + lf_out

        output_dict["prediction"] = out

        if self.use_lr_video:
            output_dict['driving_lr'] = driving_lr

        return output_dict
