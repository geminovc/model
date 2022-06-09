import torch.nn.functional as F

# source image

# optical flow generator

# warped source

# keypoints?

# loss on optical flow

#


class OverhauledGenerator():
    

    def forward(self, target):
        # get optical flow from RIFE
        imgs = [self.source, target)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        optical_flow = merged[2]

        # warp source to get warped source
        warped_source = F.grid_sample(input=source, grid=optical_flow,
                    mode='bilinear', padding_mode='border', align_corners=True)

        # get residual or what remains of target
        residual = target - warped_source #TODO: does this need normalization

        # put the source through a couple of convolutional layers
        encoded_features = self.bottleneck(self.source, optical_flow, warped_source, residual)

        # decoder side
        decoded_warp = self.decoder(encoded_features)
        encoded_reference_frame = self.reference_frame_conv(reference_frame)


        extract stuff

        return prediction

