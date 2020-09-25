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

vis = visdom.Visdom(env="ellen-vgg-prevframe")
# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

import torch.nn as nn

#from segmentation.face_parsing.model_utils import *

gpu_ids = [0]
torch.cuda.set_device(gpu_ids[0])

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_lr_paths, img_gt_paths, img_hrprev_paths, transform):
        self.img_lr_paths = img_lr_paths
        self.img_hr_paths = img_gt_paths
        self.img_hrprev_paths = img_hrprev_paths
        
        self.transform = transform
        self.seg_net = models.vgg19_bn(pretrained=True)
        self.seg_net.eval()
        
    def __getitem__(self, index):
        pre = '/nfs/disk1/video-conf/FSRNet-pytorch/'
        img_lr = cv2.imread(pre+self.img_lr_paths[index])
#         print(self.img_lr_paths[index], img_lr)
        img_hrprev = cv2.imread(pre+self.img_hrprev_paths[index])
        img_hr = cv2.imread(pre+self.img_hr_paths[index])
        
        img_lr = np.flip(img_lr, 2)
        img_hrprev = np.flip(img_hrprev, 2)
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
            img_hrprev = img_hrprev[left:right, top:bottom, :]
            img_hr = img_hr[left:right, top:bottom, :]
            
        img_lr = cv2.resize(img_lr, (256, 256))
        img_hrprev = cv2.resize(img_hrprev, (256, 256))
        img_hr = cv2.resize(img_hr, (256, 256))
        img_lr = self.transform(Image.fromarray(img_lr, 'RGB'))
        img_hrprev = self.transform(Image.fromarray(img_hrprev, 'RGB'))
        img_hr = self.transform(Image.fromarray(img_hr, 'RGB'))
        
        with torch.no_grad():
            seg_activations = self.seg_net.features[:6](img_lr.unsqueeze(0)).detach().squeeze()
        return torch.cat([img_lr, img_hrprev]), img_hr, seg_activations
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.img_lr_paths)
    
regular_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_label = transforms.Compose([
    transforms.Resize((256, 256)),
])

import pickle

pkl_fpath = '/nfs/disk1/video-conf/nets_implementation/SPADE/'
with open(pkl_fpath + 'train_lbr_ellen_season1-2_Ellen_s_Coffee_Monologue-fA9iO849Tyo_5.pkl', 'rb') as handle:
    img_lr_paths = pickle.load(handle)
    
with open(pkl_fpath + 'train_gt_ellen_season1-2_Ellen_s_Coffee_Monologue-fA9iO849Tyo_5.pkl', 'rb') as handle:
    img_gt_paths = pickle.load(handle)
    
with open(pkl_fpath + 'train_hbr_ellen_season1-2_Ellen_s_Coffee_Monologue-fA9iO849Tyo_5.pkl', 'rb') as handle:
    img_hrprev_paths = pickle.load(handle)
    
ds = Dataset(img_lr_paths, img_gt_paths, img_hrprev_paths, regular_transform)
dataloader = torch.utils.data.DataLoader(ds, batch_size=8,
    shuffle=True,
    num_workers=1, drop_last=True
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
                                   ('noisy_image', data_i[0][:, :3, :, :]), 
                                   ('prev_image', data_i[0][:, 3:, :, :]), 
                                   ])
#             visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)
            print("Synth:", visuals['synthesized_image'].shape)
            vis.images(np.clip((visuals['noisy_image'].numpy() + 1)/2, 0, 1), opts={"title":"LR"}, win="LR")
            vis.images(np.clip((visuals['prev_image'].numpy() + 1)/2, 0, 1), opts={"title":"HR_PREV"}, win="HR_PREV")
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
