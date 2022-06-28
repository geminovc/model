from torch import nn
import torch
import torch.nn.functional as F
from first_order_model.modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d

class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0, num_pixel_features=0, 
                 predict_pixel_features=False, run_at_256=False):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        self.num_kp = num_kp
        self.num_pixel_features = num_pixel_features
        
        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        if predict_pixel_features:
            self.pixel_feature_network = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=num_pixel_features * num_kp, 
                                      kernel_size=(7, 7), padding=pad)
        else:
            self.pixel_feature_network = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        self.run_at_256 = run_at_256
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3))
        kp = {'value': value}

        return kp

    def reshape(self, feature_map, final_shape, num_features, entries_per_feature, heatmap):
        """ 
        Reshape Jacobian and pixel features to right dimensions and sum
        """
        feature_map = feature_map.reshape(final_shape[0], num_features, entries_per_feature, 
                                            final_shape[2], final_shape[3])
        unsqueezed_heatmap = heatmap.unsqueeze(2)

        feature = unsqueezed_heatmap * feature_map
        feature = feature.view(final_shape[0], final_shape[1], entries_per_feature, -1)
        feature = feature.sum(dim=-1)
        return feature

    def forward(self, x):
        if x.size(dim=1) > 64:
            if self.run_at_256:
                x = F.interpolate(x, 256)
            
            if self.scale_factor != 1:
                x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian = self.reshape(jacobian_map, final_shape, 
                                    self.num_jacobian_maps, 4, heatmap)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            out['jacobian'] = jacobian

        if self.pixel_feature_network is not None:
            pixel_feature_map = self.pixel_feature_network(feature_map)
            pixel_feature = self.reshape(pixel_feature_map, final_shape, 
                                    self.num_kp, self.num_pixel_features, heatmap)
            out['pixel_features'] = pixel_feature

        return out
