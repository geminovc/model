import os
import numpy as np
import pickle
import tensorboardX
import pathlib
from torchvision import transforms



class Logger_Yaw(object):
    def __init__(self, args, experiment_dir):
        super(Logger_Yaw, self).__init__()
        self.num_iter = {('train', 'normal'): 0, ('test', 'easy_pose'): 0, ('test', 'hard_pose'): 0, ('test', 'combined_pose'): 0,
                            ('unseen_test', 'easy_pose'): 0, ('unseen_test', 'hard_pose'): 0, ('unseen_test', 'combined_pose'): 0} 
        
        self.no_disk_write_ops = args.no_disk_write_ops
        self.rank = args.rank

        if not self.no_disk_write_ops:
            self.experiment_dir = experiment_dir

            # Directory to save train data
            os.makedirs(experiment_dir / 'images' / 'train', exist_ok=True)
            
            # Directory to save test and unseen_test data
            for phase in ['test', 'unseen_test']:
                for pose_component in ['easy_pose', 'hard_pose', 'combined_pose']:
                    os.makedirs(experiment_dir / 'images' / phase / pose_component, exist_ok=True)
            
            
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
            self.num_iter[(phase, pose_component)] += 1

            # Save visuals
            # If you're saving metrics save them at a different folder depending on index so that
            # all images in a folder are the same so you can compare them accross runs easily
            if phase != 'train':
                self.to_image(visuals).save(self.experiment_dir \
                        / 'images' / phase / str(pose_component) / ('%04d_%02d.jpg' % (self.num_iter[(phase, pose_component)], self.rank)))
            else:
                self.to_image(visuals).save(self.experiment_dir \
                        / 'images' / phase / ('%04d_%02d.jpg' % (self.num_iter[(phase, pose_component)], self.rank)))

            if self.rank != 0:
                return

            # Sets tensorboard_phase to metrics_1/2/3/ etc if you're in metrivs in order to separate the 
            # different metrics images in the tensorboard
            if phase != 'train':
                tensorboard_phase = f'{phase}_{pose_component}'
            else:
                tensorboard_phase = phase
        
            self.writer.add_image(f'results_{tensorboard_phase}', visuals, self.num_iter[(phase, pose_component)])

            # Save losses
            for key, value in losses.items():
                if key in self.losses:
                    self.losses[key].append(value)
                else:
                    self.losses[key] = [value]
                self.writer.add_scalar(f'losses/{key}_{tensorboard_phase}', value, self.num_iter[(phase, pose_component)])

            for key, value in metrics.items():
                if key in self.metrics:
                    self.metrics[key].append(value)
                else:
                    self.metrics[key] = [value]
                self.writer.add_scalar(f'metrics/{key}_{tensorboard_phase}', value, self.num_iter[(phase, pose_component)])
            
            # Save losses and metrics
            pickle.dump(self.losses,  open(str(self.experiment_dir) + "/" + 'losses_'  + str(phase) + "_" + str(pose_component) + '.pkl', 'wb'))
            pickle.dump(self.metrics, open(str(self.experiment_dir) + "/" + 'metrics_' + str(phase) + "_" + str(pose_component) + '.pkl', 'wb'))

        elif self.rank != 0:
            return

        # Print losses and metrics
        print(phase, pose_component, 'losses:', ', '.join('%s: %.3f' % (key, value) for key, value in losses.items()) + ', time: %.3f' % time)
        print(phase, pose_component,  'metrics:', ', '.join('%s: %.3f' % (key, value) for key, value in metrics.items()) + ', time: %.3f' % time)

    def set_num_iter(self, train_iter, test_easy_pose_iter, test_hard_pose_iter, test_combined_pose_iter, 
                unseen_test_easy_pose_iter, unseen_test_hard_pose_iter, unseen_test_combined_pose_iter):

        self.num_iter = {('train', 'normal'): train_iter, ('test', 'easy_pose'): test_easy_pose_iter, ('test', 'hard_pose'): test_hard_pose_iter,
         ('test', 'combined_pose'): test_combined_pose_iter,('unseen_test', 'easy_pose'): unseen_test_easy_pose_iter,
         ('unseen_test', 'hard_pose'): unseen_test_hard_pose_iter, ('unseen_test', 'combined_pose'): unseen_test_combined_pose_iter} 
