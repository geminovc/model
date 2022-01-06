import torch
from torch import nn
import torch.nn.functional as F
from first_order_model.modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from first_order_model.onnx.modules.dense_motion import DenseMotionNetwork_ONNX
from mmcv.ops.point_sample import bilinear_grid_sample


class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork_ONNX(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        # return F.grid_sample(inp, deformation), deformation
        return bilinear_grid_sample(inp, deformation), deformation

    def forward(self, source_image, kp_driving_v, kp_driving_j, kp_source_v, kp_source_j):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion_m, dense_motion_sparse_deformed, dense_motion_occ_map, dense_motion_deformation = self.dense_motion_network(source_image=source_image,
                                                     kp_driving_v=kp_driving_v, kp_driving_j=kp_driving_j,
                                                     kp_source_v=kp_source_v, kp_source_j=kp_source_j)
            output_dict_m = dense_motion_m
            output_dict_sparse_deformed = dense_motion_sparse_deformed
            output_dict_occ_m = dense_motion_occ_map
            deformation = dense_motion_deformation
            occlusion_map = dense_motion_occ_map

            out, _ = self.deform_input(out, deformation)

            if output_dict_occ_m is not None:
                if out.shape[2] != output_dict_occ_m.shape[2] or out.shape[3] != output_dict_occ_m.shape[3]:
                    occlusion_map = F.interpolate(output_dict_occ_m, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

            output_dict_deformed, deformation = self.deform_input(source_image, deformation)
            output_dict_deformation = deformation

        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict_prediction = out

        # return (output_dict_prediction, output_dict_m, output_dict_deformation, output_dict_deformed, output_dict_occ_m, output_dict_sparse_deformed)
        return output_dict_prediction