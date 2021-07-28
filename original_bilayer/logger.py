'''
This script is used to as the logging module. 

Arguments
----------

experiment_dir: The directory to save the logs in

differentiate_by_poses: boolean variable, if True, the logging information contains pose distribution information namely the information 
about the losses and metrics of test/unseen_test data of hard/easy/combined poses 

no_disk_write_ops: If True, the logs are not saved on the disk and only tensorboard information is saved.  

Outputs
----------
Saves the visuals in jpg, metrics and losses as pickle files, and tensorboard values on the disk.  


'''

import os
import numpy as np
import pickle
import tensorboardX
import pathlib
from torchvision import transforms



class Logger(object):
    def __init__(self, args, experiment_dir, differentiate_by_poses):
        super(Logger, self).__init__()
        
        self.differentiate_by_poses = differentiate_by_poses
        if self.differentiate_by_poses:
            self.num_iter = {('train', 'none'): 0, ('metrics', 'none'): 0, ('test', 'easy_pose'): 0, ('test', 'hard_pose'): 0, ('test', 'combined_pose'): 0,
                    ('unseen_test', 'easy_pose'): 0, ('unseen_test', 'hard_pose'): 0, ('unseen_test', 'combined_pose'): 0} 
        else:
            self.num_iter = {'train': 0, 'test': 0, 'metrics' : 0, 'unseen_test': 0} # Added metrics set to 0
        
        self.no_disk_write_ops = args.no_disk_write_ops
        self.rank = args.rank

        # Making directories on the disk to save data if no_disk_write_ops is Flase
        if not self.no_disk_write_ops:
            self.experiment_dir = experiment_dir

            os.makedirs(experiment_dir / 'images' / 'train', exist_ok=True)
           
            # Create directory structure for test and unseen_test by subdivided pose information
            if self.differentiate_by_poses:
                for phase in ['test', 'unseen_test']:
                    for pose_component in ['easy_pose', 'hard_pose', 'combined_pose']:
                        os.makedirs(experiment_dir / 'images' / phase / pose_component, exist_ok=True)
            
            else:
                for phase in ['test', 'unseen_test']:
                    os.makedirs(experiment_dir / 'images' / phase, exist_ok=True)
            
            # Do not make metrics subfolders if skip_metrics is False
            if not args.skip_metrics:
                os.makedirs(experiment_dir / 'images' / 'metrics', exist_ok=True)
                for index in range(1, args.num_metrics_images + 1):
                    os.makedirs(experiment_dir / 'images' / 'metrics' / str(index), exist_ok=True)
            
            self.to_image = transforms.ToPILImage()

            if args.rank == 0:
                if args.which_epoch != 'none' and args.init_experiment_dir == '':
                    self.losses = pickle.load(open(self.experiment_dir / 'losses.pkl', 'rb')) 
                    self.losses = pickle.load(open(self.experiment_dir / 'metrics.pkl', 'rb'))
                else:
                    self.losses = {}
                    self.metrics = {}
                self.writer = tensorboardX.SummaryWriter(args.experiment_dir + '/runs/' + args.experiment_name + '/tensorboard/')
                
    def output_logs(self, phase, pose_component, visuals, losses, metrics, time, metrics_index=None):
        
        # Allows you to separate out the metrics on the tensorboards for easy viewing
        if not self.no_disk_write_ops:
            # Increment iter counter
            if (phase != 'metrics' or (phase == 'metrics' and metrics_index == 1)):
                if self.differentiate_by_poses:
                    self.num_iter[(phase, pose_component)] += 1
                else:
                    self.num_iter[phase] += 1
                
            # Save visuals
            # If you're saving metrics save them at a different folder depending on index so that
            # All images in a folder are the same so you can compare them accross runs easily
            
            if self.differentiate_by_poses:
                if phase != 'train':
                    if phase == 'metrics':
                        self.to_image(visuals).save(self.experiment_dir \
                            / 'images' / phase / str(metrics_index) / ('%04d_%02d.jpg' % (self.num_iter[(phase, pose_component)], self.rank)))
                    else:
                        self.to_image(visuals).save(self.experiment_dir \
                            / 'images' / phase / str(pose_component) / ('%04d_%02d.jpg' % (self.num_iter[(phase, pose_component)], self.rank)))

                else:
                    self.to_image(visuals).save(self.experiment_dir \
                        / 'images' / phase / ('%04d_%02d.jpg' % (self.num_iter[(phase, pose_component)], self.rank)))
                
            else:
                if phase == 'metrics':
                    self.to_image(visuals).save(self.experiment_dir \
                            / 'images' / phase / str(metrics_index) / ('%04d_%02d.jpg' % (self.num_iter[phase], self.rank)))
                else:
                    self.to_image(visuals).save(self.experiment_dir \
                            / 'images' / phase / ('%04d_%02d.jpg' % (self.num_iter[phase], self.rank)))

            if self.rank != 0:
                return

            # Sets tensorboard_phase to metrics_1/2/3/ etc if you're in metrics in order to separate the 
            # different metrics images in the tensorboard
            
            if self.differentiate_by_poses:
                if phase != 'train':
                    tensorboard_phase = f'{phase}_{pose_component}'
                elif phase == 'metrics':
                    tensorboard_phase = f'{phase}_{metrics_index}'
                else:
                    tensorboard_phase = phase

                self.writer.add_image(f'results_{tensorboard_phase}', visuals, self.num_iter[(phase, pose_component)])
            else:

                if phase == 'metrics':
                    tensorboard_phase = f'{phase}_{metrics_index}'
                else:
                    tensorboard_phase = phase
            
                self.writer.add_image(f'results_{tensorboard_phase}', visuals, self.num_iter[phase])

            # Save losses, metrics, and tensorboard 
            for key, value in losses.items():
                if key in self.losses:
                    self.losses[key].append(value)
                else:
                    self.losses[key] = [value]
                if self.differentiate_by_poses:
                    self.writer.add_scalar(f'losses/{key}_{tensorboard_phase}', value, self.num_iter[(phase, pose_component)])
                else:
                    self.writer.add_scalar(f'losses/{key}_{tensorboard_phase}', value, self.num_iter[phase])

            for key, value in metrics.items():
                if key in self.metrics:
                    self.metrics[key].append(value)
                else:
                    self.metrics[key] = [value]
                                    
                if self.differentiate_by_poses:
                    self.writer.add_scalar(f'metrics/{key}_{tensorboard_phase}', value, self.num_iter[(phase, pose_component)])
                else:
                    self.writer.add_scalar(f'metrics/{key}_{tensorboard_phase}', value, self.num_iter[phase])
            
            # Save losses and metrics
            pickle.dump(self.losses,  open(str(self.experiment_dir) + "/" + 'losses_'  + str(tensorboard_phase) + '.pkl', 'wb'))
            pickle.dump(self.metrics, open(str(self.experiment_dir) + "/" + 'metrics_' + str(tensorboard_phase) + '.pkl', 'wb'))

        elif self.rank != 0:
            return

        # Print losses and metrics
        print(phase, pose_component, 'losses:', ', '.join('%s: %.3f' % (key, value) for key, value in losses.items()) + ', time: %.3f' % time)
        print(phase, pose_component, 'metrics:', ', '.join('%s: %.3f' % (key, value) for key, value in metrics.items()) + ', time: %.3f' % time)

    def set_num_iter_no_pose(self, train_iter, test_iter, metrics_iter, unseen_test_iter):
        self.num_iter = {
            'train': train_iter,
            'test': test_iter,
            'metrics' : metrics_iter,
            'unseen_test': unseen_test_iter}

    def set_num_iter_with_pose(self, train_iter, metrics_iter, test_easy_pose_iter, test_hard_pose_iter, test_combined_pose_iter, 
                unseen_test_easy_pose_iter, unseen_test_hard_pose_iter, unseen_test_combined_pose_iter):

        self.num_iter = {('train', 'none'): train_iter, ('metrics', 'none'): metrics_iter, ('test', 'easy_pose'): test_easy_pose_iter, 
        ('test', 'hard_pose'): test_hard_pose_iter, ('test', 'combined_pose'): test_combined_pose_iter,('unseen_test', 'easy_pose'): unseen_test_easy_pose_iter,
         ('unseen_test', 'hard_pose'): unseen_test_hard_pose_iter, ('unseen_test', 'combined_pose'): unseen_test_combined_pose_iter} 
