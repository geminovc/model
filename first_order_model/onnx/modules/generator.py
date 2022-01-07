import torch
from torch import nn
import torch.nn.functional as F
from first_order_model.modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from first_order_model.modules.generator import OcclusionAwareGenerator


class OcclusionAwareGenerator_ONNX(OcclusionAwareGenerator):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def forward(self, source_image, kp_driving_v, kp_driving_j, kp_source_v, kp_source_j):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion_occ_map, dense_motion_deformation = self.dense_motion_network(source_image=source_image,
                                                     kp_driving_v=kp_driving_v, kp_driving_j=kp_driving_j,
                                                     kp_source_v=kp_source_v, kp_source_j=kp_source_j)
            output_dict_occ_m = dense_motion_occ_map
            deformation = dense_motion_deformation
            occlusion_map = dense_motion_occ_map

            out, _ = self.deform_input(out, deformation)

            if output_dict_occ_m is not None:
                if out.shape[2] != output_dict_occ_m.shape[2] or out.shape[3] != output_dict_occ_m.shape[3]:
                    occlusion_map = F.interpolate(output_dict_occ_m, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict_prediction = out

        return output_dict_prediction