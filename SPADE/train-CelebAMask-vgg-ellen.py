"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import os
from PIL import Image
import cv2
import numpy as np
import visdom
cv2.setNumThreads(0)

vis = visdom.Visdom(env="ellen-vgg")
# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

import torch.nn as nn

from segmentation.face_parsing.model_utils import *


class unet(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=19,
        is_deconv=True,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
#         print(up2.shape)
#         final = self.final(up1)
        return up1


gpu_ids = [0]
torch.cuda.set_device(gpu_ids[0])

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_lr_paths, img_hr_paths, transform):
        self.img_lr_paths = img_lr_paths
        self.img_hr_paths = img_hr_paths
        self.transform = transform
        self.seg_net = models.vgg19_bn(pretrained=True)
#         self.seg_net.load_state_dict(torch.load('./segmentation/face_parsing/models/parsenet/447250_G.pth'))
        self.seg_net.eval()
        
    def __getitem__(self, index):
#         print(self.img_lr_paths[index])
#         print("File:", img_lr_paths[index])
        pre = '/nfs/disk1/video-conf/FSRNet-pytorch/'
        img_lr = cv2.imread(pre+self.img_lr_paths[index])
        img_hr = cv2.imread(pre+self.img_hr_paths[index])
        if img_lr is None or img_hr is None:
            img_lr = cv2.imread(self.img_lr_paths[0])
            img_hr = cv2.imread(self.img_hr_paths[0])
        img_lr = np.flip(img_lr, 2)
        img_hr = np.flip(img_hr, 2)
        
        if img_lr.shape[0] != img_lr.shape[1]:
            width, height, _ = img_lr.shape   # Get dimensions
            min_len = min(width, height)
            new_width, new_height = min_len, min_len
            left = int((width - new_width)/2)
            top = int((height - new_height)/2)
            right = int((width + new_width)/2)
            bottom = int((height + new_height)/2)
            
            img_lr = img_lr[left:right, top:bottom, :]
            img_hr = img_hr[left:right, top:bottom, :]
            
        img_lr = cv2.resize(img_lr, (256, 256))
        img_hr = cv2.resize(img_hr, (256, 256))
        img_lr = self.transform(Image.fromarray(img_lr, 'RGB'))
        img_hr = self.transform(Image.fromarray(img_hr, 'RGB'))
        
        with torch.no_grad():
            seg_activations = self.seg_net.features[:6](img_lr.unsqueeze(0)).detach().squeeze()
        return img_lr, img_hr, seg_activations
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.img_lr_paths)
    
regular_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_label = transforms.Compose([
    transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize((0, 0, 0), (0, 0, 0)),
])

import pickle

with open('/nfs/disk1/video-conf/FSRNet-pytorch/ellen_lr_train', 'rb') as handle:
    img_lr_paths = pickle.load(handle)
    
with open('/nfs/disk1/video-conf/FSRNet-pytorch/ellen_hr_train', 'rb') as handle:
    img_hr_paths = pickle.load(handle)
    
ds = Dataset(img_lr_paths, img_hr_paths, regular_transform)
dataloader = torch.utils.data.DataLoader(ds, batch_size=8,
    shuffle=True,
    num_workers=32, drop_last=True
)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

# segmentation

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
#         break
        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i[2]),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data_i[1]),
                                   ('noisy_image', data_i[0])])
#             visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)
            print("Synth:", visuals['synthesized_image'].shape)
            vis.images(np.clip((visuals['noisy_image'].numpy() + 1)/2, 0, 1), opts={"title":"Noisy"}, win="noisy")
            vis.images((visuals['real_image'].numpy() + 1)/2, opts={"title":"GT"}, win="gt")
            vis.images((visuals['synthesized_image'].cpu().detach().numpy() + 1)/2, opts={"title":"Generated Output"}, win="generated output")
        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
