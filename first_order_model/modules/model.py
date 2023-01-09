from torch import nn
import torch
import torch.nn.functional as F
from first_order_model.modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from torchvision import models
import numpy as np
from torch.autograd import grad

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

    def compute_loss(self, X, Y, weights=[10, 10, 10, 10, 10]):
        X_vgg = self.forward(X)
        Y_vgg = self.forward(Y)

        loss_val = 0
        diffs = [(x - y)**2 for x, y in zip(X_vgg, Y_vgg)]
        for d, w in zip(diffs, weights):
            loss_val += w * d.mean()
        loss_val /= np.sum(weights)
        return loss_val


class VggFace16(torch.nn.Module):
    """
    Vgg16 network for face perceptual loss. Was added by Vibhaa.
    """
    def __init__(self, requires_grad=False):
        super(VggFace16, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 26):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([103.939/127.5, 116.779/127.5, \
                                        123.680/127.5]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([1/127.5, 1/127.5, 1/127.5]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

    def compute_loss(self, X, Y, weights=[10, 10, 10, 10, 10]):
        X_vgg = self.forward(X)
        Y_vgg = self.forward(Y)

        loss_val = 0
        diffs = [(x - y)**2 for x, y in zip(X_vgg, Y_vgg)]
        for d, w in zip(diffs, weights):
            loss_val += w * d.mean()
        loss_val /= np.sum(weights)
        return loss_val


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()
        
        if self.train_params.get('conditional_gan', False):
            self.cgan_pyramide = ImagePyramide(self.scales, 2 * generator.num_channels)
            if torch.cuda.is_available():
                self.cgan_pyramide = self.cgan_pyramide.cuda()

        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

        if sum(self.loss_weights.get('perceptual_face', [0])) != 0:
            self.vgg_face = VggFace16()
            if torch.cuda.is_available():
                self.vgg_face = self.vgg_face.cuda()


    def forward(self, x, generator_type='occlusion_aware'):
        driving_lr =  x.get('driving_lr', None)

        if generator_type in ['occlusion_aware', 'split_hf_lf']:
            kp_source = self.kp_extractor(x['source'])
            
            if driving_lr is not None:
                kp_driving = self.kp_extractor(driving_lr)
            else:
                kp_driving = self.kp_extractor(x['driving'])
            
            generated = self.generator(x['source'], kp_source=kp_source, 
                    kp_driving=kp_driving, update_source=True, 
                    driving_lr=driving_lr)
            generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
        else:
            assert(driving_lr is not None)
            generated = self.generator(driving_lr)
            generated.update({'driving_lr': driving_lr})
        
        loss_values = {}

        # standard pyramides for Vgg perceptual loss
        if 'teacher' in x:
            print('using teacher output for losses')
        real_input = x['driving'] if 'teacher' not in x else x['teacher']
        generated_input = generated['prediction']
        pyramide_real = self.pyramid(real_input)
        pyramide_generated = self.pyramid(generated_input)
        
        # pyramides for conditional gan if need be to be used by discriminator
        if self.train_params.get('conditional_gan', False):
            concatenated_real_input = torch.cat([real_input, x['source']], dim=1)
            concatenated_generated_input = torch.cat([generated_input, x['source']], dim=1)
            disc_pyramide_real = self.cgan_pyramide(concatenated_real_input)
            disc_pyramide_generated = self.cgan_pyramide(concatenated_generated_input)
        else:
            disc_pyramide_real = pyramide_real
            disc_pyramide_generated = pyramide_generated
        
        # use only HF pipeline for perceptual if there's a split
        if generator_type == 'split_hf_lf':
            generated_input_lf_detached = generated['prediction_lf_detached']
            pyramide_generated_lf_detached = self.pyramid(generated_input_lf_detached)
            pyramid_generated = pyramide_generated_lf_detached
        
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if sum(self.loss_weights.get('perceptual_face', [0])) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg_face(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg_face(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual_face']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += weight * value
                loss_values['perceptual_face'] = value_total

        if self.loss_weights.get('pixelwise', 0) != 0:
            loss_dict = {
                            'mse': F.mse_loss,
                            'l1': F.l1_loss,
                            'ce': F.cross_entropy
                        }
            loss_fn = loss_dict['l1']
            generated_lf = generated['prediction_lf'] if generator_type == 'split_hf_lf' \
                    else generated['prediction']
            pix_loss = loss_fn(generated_lf, real_input.detach())
            loss_values['pixelwise'] = self.loss_weights['pixelwise'] * pix_loss
                   
        if self.loss_weights['generator_gan'] != 0:
            if generator_type == 'occlusion_aware':
                discriminator_maps_generated = self.discriminator(disc_pyramide_generated, kp=detach_kp(kp_driving))
                discriminator_maps_real = self.discriminator(disc_pyramide_real, kp=detach_kp(kp_driving))
            else:
                discriminator_maps_generated = self.discriminator(disc_pyramide_generated)
                discriminator_maps_real = self.discriminator(disc_pyramide_real)

            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0 and transformed_kp['value'].requires_grad:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                                    transformed_kp['jacobian'])

                normed_driving = torch.inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales

        channel_offset = 2 if self.train_params.get('conditional_gan', False) else 1
        self.pyramid = ImagePyramide(self.scales, channel_offset * generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        real_input = x['driving']
        generated_input = generated['prediction'].detach()
        if self.train_params.get('conditional_gan', False):
            real_input = torch.cat([real_input, x['source']], dim=1)
            generated_input = torch.cat([generated_input, x['source']], dim=1)
        
        pyramide_real = self.pyramid(real_input)
        pyramide_generated = self.pyramid(generated_input)

        if 'kp_driving' in generated:
            kp_driving = generated['kp_driving']
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
        else:
            discriminator_maps_generated = self.discriminator(pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values
