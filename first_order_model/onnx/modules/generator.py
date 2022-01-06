import torch
from torch import nn
import torch.nn.functional as F
from first_order_model.modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from first_order_model.modules.generator import OcclusionAwareGenerator
from mmcv.ops.point_sample import bilinear_grid_sample


class OcclusionAwareGenerator_ONNX(OcclusionAwareGenerator):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

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