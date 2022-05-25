from torch import nn
import torch.nn.functional as F
import torch
from first_order_model.modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian
from first_order_model.modules.dense_motion import DenseMotionNetwork


class DenseMotionNetwork_ONNX(DenseMotionNetwork):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def create_heatmap_representations(self, source_image, kp_driving_v, kp_driving_j, kp_source_v, kp_source_j):
        """
        Eq 6. in the paper H_k(z)
        Different inputs from the original version
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving_v, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source_v, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        #adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    def create_sparse_motions(self, source_image, kp_driving_v, kp_driving_j, kp_source_v, kp_source_j):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source_v.type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - kp_driving_v.view(bs, self.num_kp, 1, 1, 2)
        # TODO: Add Jacobians if found a solution for torch.inverse conversion in onnx
        driving_to_source = coordinate_grid + kp_source_v.view(bs, self.num_kp, 1, 1, 2)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions

    def forward(self, source_image, kp_driving_v, kp_driving_j, kp_source_v, kp_source_j):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving_v, kp_driving_j, kp_source_v, kp_source_j)
        sparse_motion = self.create_sparse_motions(source_image, kp_driving_v, kp_driving_j, kp_source_v, kp_source_j)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        out_dict_sparse_deformed = deformed_source

        input = torch.cat([heatmap_representation, deformed_source], dim=2)
        input = input.view(bs, -1, h, w)

        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict_mask = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict_deformation = deformation

        # Sec. 3.2 in the paper
        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict_occlusion_map = occlusion_map
        else:
            out_dict_occlusion_map = None
        return out_dict_occlusion_map, out_dict_deformation
