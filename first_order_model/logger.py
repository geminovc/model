import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections
import tensorboardX
import flow_vis
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import piq
import math


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None
        self.writer = tensorboardX.SummaryWriter(os.path.join(log_dir, 'tensorboard'))
        self.metrics_averages = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()
        
        for name, value in zip(loss_names, loss_mean):
            self.writer.add_scalar(f'losses/{name}', value, self.epoch)


    """ get visual metrics for the model's reconstruction """
    @staticmethod
    def get_visual_metrics(prediction, original, loss_fn_vgg, original_lpips, face_lpips):
        if torch.cuda.is_available():
            original = original.cuda()
            prediction = prediction.cuda()
        lpips_val = loss_fn_vgg(original, prediction).data.cpu().numpy().flatten()[0]
        face_lpips_val = face_lpips(original, prediction).data.cpu().numpy().flatten()[0]
        original_lpips_val = original_lpips(original, prediction).data.cpu().numpy().flatten()[0]
        
        ssim = piq.ssim(original, prediction, data_range=1.).data.cpu().numpy().flatten()[0]
        ssim_db = -20 * math.log10(1 - ssim)
        psnr = piq.psnr(original, prediction, data_range=1., reduction='none').data.cpu().numpy()
        
        return {'psnr': psnr, 'ssim': ssim, 'lpips': lpips_val, 'ssim_db': ssim_db, \
                'orig_lpips': original_lpips_val, 'face_lpips': face_lpips_val}

    def log_metrics_images(self, iteration, input_data, output, loss_fn_vgg, original_lpips, face_lpips):
        if iteration == 0:
            if self.metrics_averages is not None:
                for name, values in self.metrics_averages.items():
                    average = np.mean(values)
                    self.writer.add_scalar(f'metrics/{name}', average, self.epoch)
            self.metrics_averages = {'psnr': [], 'ssim': [], 'lpips': [], 'ssim_db': [], \
                    'orig_lpips': [], 'face_lpips':[]}

        image = self.visualizer.visualize(input_data['driving'], input_data['source'], output)
        self.writer.add_image(f'metrics{iteration}', image, self.epoch, dataformats='HWC')
        metrics = Logger.get_visual_metrics(output['prediction'], input_data['driving'],\
                loss_fn_vgg, original_lpips, face_lpips)
        for name, value in metrics.items():
            self.metrics_averages[name].append(value)

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, 
                "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)
        self.writer.add_image('reconstruction', image, self.epoch, dataformats='HWC')

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items() if v is not None}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None,
                 optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None, 
                 device='gpu', dense_motion_network=None, upsampling_enabled=False, use_lr_video=[], 
                 hr_skip_connections=False, run_at_256=True, generator_type='occlusion_aware', reconstruction=False):

        if device == torch.device('cpu'):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)

        if reconstruction:
            print("loading everything in generator as is")
            generator.load_state_dict(checkpoint['generator'])

        elif generator is None and dense_motion_network is not None:
            gen_params = checkpoint['generator']
            dense_motion_params = {k: gen_params[k] for k in gen_params.keys() if k.startswith('dense_motion_network')}
            generator.load_state_dict(dense_motion_params, strict=False)
            print("loading only dense motion in generator")
        elif generator is not None and upsampling_enabled:
            if hr_skip_connections or run_at_256:
                # skip connections used in the decoder or bring everything down to same dimensions 
                # as original FOMM pipeline
                modified_generator_params = {k: v for k, v in checkpoint['generator'].items() \
                    if not (k.startswith("final") or k.startswith("sigmoid"))}
                print("loading everything in generator except final and sigmoid")
            else:
                modified_generator_params = {k: v for k, v in checkpoint['generator'].items() \
                    if not (k.startswith("final") or k.startswith("sigmoid") or k.startswith('first'))}
                print("loading everything in generator except final and sigmoid and first")

            if len(use_lr_video) > 0 and generator_type == 'occlusion_aware':
                if 'decoder' in use_lr_video:
                    modified_generator_params = {k: v for k, v in modified_generator_params.items() \
                        if not k.startswith("up_blocks")}
                    print("not loading upblocks in generator")
                
                if 'hourglass_input' in use_lr_video:
                    modified_generator_params = {k: v for k, v in modified_generator_params.items() \
                        if not (k.startswith("dense_motion_network.hourglass") or \
                        k.startswith("dense_motion_network.mask") or \
                        k.startswith("dense_motion_network.occlusion"))}
                    print("not loading hourglass or mask or occlusion blocks")
                
                if 'hourglass_output' in use_lr_video:
                    modified_generator_params = {k: v for k, v in modified_generator_params.items() \
                        if not (k.startswith("dense_motion_network.mask") or \
                        k.startswith("dense_motion_network.occlusion"))}
                    print("not loading hourglass or mask or occlusion blocks")

                if 'hourglass_input' in use_lr_video and 'decoder' in use_lr_video:
                    modified_generator_params = {k: v for k, v in modified_generator_params.items() \
                        if not k.startswith("bottleneck")}
                    print("not loading bottleneck")

            generator.load_state_dict(modified_generator_params, strict=False)
        
        elif generator is not None and dense_motion_network is None and generator_type == 'occlusion_aware':
            gen_params = checkpoint['generator']
            gen_params_but_dense_motion_params = {k: gen_params[k] for k in gen_params.keys() if not k.startswith('dense_motion_network')}
            generator.load_state_dict(gen_params_but_dense_motion_params, strict=False)
            print("loading everything but dense motion in generator")
        elif generator is not None and generator_type in ['occlusion_aware', 'split_hf_lf']:
            print("loading everything in generator as is")
            generator.load_state_dict(checkpoint['generator'])
        elif generator is not None and generator_type == "super_resolution":
            modified_generator_params = {k: v for k, v in checkpoint['generator'].items() \
                if (k.startswith("bottleneck") or k.startswith("up"))}
            generator.load_state_dict(modified_generator_params, strict=False)
            print("SR: loading bottleneck and upblocks, not loading final/first because of dimensions")

        if kp_detector is not None:
            print("loading everything in kp detector as is")
            kp_detector.load_state_dict(checkpoint['kp_detector'])

        if discriminator is not None: 
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
               print("loading everything in discriminator as is")
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')

        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
            print("loading everything in generator optimizer as is")

        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
                print("loading everything in discriminator optimizer as is")
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            print("loading everything in kp detector optimizer as is")
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])

        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        if self.epoch > 0:
            self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)
        self.identity_grid = None

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def draw_deformation_heatmap(self, deformation):
        b, h, w = deformation.shape[0:3]
        if self.identity_grid is None:
            self.identity_grid = np.zeros((h, w, 2))
            for i, ival in enumerate(np.linspace(-1, 1, h)):
                for j, jval in enumerate(np.linspace(-1, 1, w)):
                    self.identity_grid[i][j][0] = jval
                    self.identity_grid[i][j][1] = ival

        deformation_heatmap = np.zeros((b, h, w, 3))
        for i in range(b):
            deformation_heatmap[i] = flow_vis.flow_to_color(deformation[i] - self.identity_grid)
        return deformation_heatmap


    def visualize(self, driving, source, out):
        images = []

        # Source image with keypoints
        source = source.data.cpu()
        source = np.transpose(source, [0, 2, 3, 1])
        if 'kp_source' in out:
            kp_source = out['kp_source']['value'].data.cpu().numpy()
            images.append((source, kp_source))

        # Equivariance visualization
        if 'transformed_frame' in out:
            transformed = out['transformed_frame'].data.cpu().numpy()
            transformed = np.transpose(transformed, [0, 2, 3, 1])
            transformed_kp = out['transformed_kp']['value'].data.cpu().numpy()
            images.append((transformed, transformed_kp))

        #LR driving image
        if 'driving_lr' in out:
            driving_lr = out['driving_lr']
            size = driving_lr.shape[2]
            driving_lr_padded = torch.zeros(driving.shape[0], 3, driving.shape[2], driving.shape[3]) 
            driving_lr_padded[:, :, :size, :size] = driving_lr
            driving_lr_padded = driving_lr_padded.data.cpu().numpy()
            driving_lr_padded = np.transpose(driving_lr_padded, [0, 2, 3, 1])
            images.append(driving_lr_padded)
       
        # Driving image with keypoints
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        if 'kp_driving' in out:
            kp_driving = out['kp_driving']['value'].data.cpu().numpy()
            images.append((driving, kp_driving))
        images.append(driving)


        # Deformed image
        if 'deformed' in out:
            deformed = out['deformed'].data.cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)
        
        # deformation heatmap
        if 'deformation' in out:
            deformation = out['deformation'].data.cpu().numpy()
            heatmap = self.draw_deformation_heatmap(deformation)
            images.append(heatmap)

        # show the lf and hf separately
        if 'prediction_lf' in out and 'prediction_hf' in out:
            prediction_lf = out['prediction_lf'].data.cpu().numpy()
            prediction_lf = np.transpose(prediction_lf, [0, 2, 3, 1])
            images.append(prediction_lf)
        
            prediction_hf = out['prediction_hf'].data.cpu().numpy()
            prediction_hf = np.transpose(prediction_hf, [0, 2, 3, 1])
            images.append(prediction_hf)

        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['value'].data.cpu().numpy()
            images.append((prediction, kp_norm))
        images.append(prediction)

        # residual/difference
        residual = np.zeros(driving.shape)
        for c, (d, p) in enumerate(zip(driving, prediction)):
            diff = d - p
            diff = (diff + np.ones_like(d)) / 2.0
            residual[c] = diff
        images.append(residual)

        # Occlusion map
        if 'occlusion_map' in out:
            occlusion_map = out['occlusion_map'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)
        
        if 'lr_occlusion_map' in out:
            occlusion_map = out['lr_occlusion_map'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)
        
        if 'hr_background_mask' in out:
            occlusion_map = out['hr_background_mask'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)

        # Deformed images according to each individual transform
        if 'sparse_deformed' in out:
            full_mask = []
            for i in range(out['sparse_deformed'].shape[1]):
                image = out['sparse_deformed'][:, i].data.cpu()
                image = F.interpolate(image, size=source.shape[1:3])
                mask = out['mask'][:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)
                mask = F.interpolate(mask, size=source.shape[1:3])
                image = np.transpose(image.numpy(), (0, 2, 3, 1))
                mask = np.transpose(mask.numpy(), (0, 2, 3, 1))

                if i != 0:
                    color = np.array(self.colormap((i - 1) / (out['sparse_deformed'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 1, 3))

                images.append(image)
                if i != 0:
                    images.append(mask * color)
                else:
                    images.append(mask)

                full_mask.append(mask * color)

            images.append(sum(full_mask))

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
