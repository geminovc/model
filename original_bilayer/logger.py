import os
import numpy as np
import pickle
import tensorboardX
import pathlib
from torchvision import transforms



class Logger(object):
    def __init__(self, args, experiment_dir):
        super(Logger, self).__init__()
        self.num_iter = {'train': 0, 'test': 0}
        
        self.no_disk_write_ops = args.no_disk_write_ops
        self.rank = args.rank

        if not self.no_disk_write_ops:
            self.experiment_dir = experiment_dir

            for phase in ['train', 'test']:
                os.makedirs(experiment_dir / 'images' / phase, exist_ok=True)

            self.to_image = transforms.ToPILImage()

            if args.rank == 0:
                if args.which_epoch != 'none' and args.init_experiment_dir == '':
                    self.losses = pickle.load(open(self.experiment_dir / 'losses.pkl', 'rb')) 
                    self.losses = pickle.load(open(self.experiment_dir / 'metrics.pkl', 'rb'))
                else:
                    self.losses = {}
                    self.metrics = {}
                self.writer = tensorboardX.SummaryWriter(args.experiment_dir + '/runs/' + args.experiment_name + '/tensorboard_paper/')
                
    def output_logs(self, phase, visuals, losses, metrics, time):
        if not self.no_disk_write_ops:
            # Increment iter counter
            self.num_iter[phase] += 1

            # Save visuals
            self.to_image(visuals).save(self.experiment_dir \
                    / 'images' / phase / ('%04d_%02d.jpg' % (self.num_iter[phase], self.rank)))

            if self.rank != 0:
                return

            self.writer.add_image(f'results_{phase}', visuals, self.num_iter[phase])

            # Save losses
            for key, value in losses.items():
                if key in self.losses:
                    self.losses[key].append(value)
                else:
                    self.losses[key] = [value]
                self.writer.add_scalar(f'losses_{key}_{phase}', value, self.num_iter[phase])

            for key, value in metrics.items():
                if key in self.metrics:
                    self.metrics[key].append(value)
                else:
                    self.metrics[key] = [value]
                self.writer.add_scalar(f'metrics_{key}_{phase}', value, self.num_iter[phase])
            
            # Save losses and metrics
            pickle.dump(self.losses, open(self.experiment_dir / 'losses.pkl', 'wb'))
            pickle.dump(self.metrics, open(self.experiment_dir / 'metrics.pkl', 'wb'))

        elif self.rank != 0:
            return

        # Print losses and metrics
        print(phase, 'losses:', ', '.join('%s: %.3f' % (key, value) for key, value in losses.items()) + ', time: %.3f' % time)
        print(phase, 'metrics:', ', '.join('%s: %.3f' % (key, value) for key, value in metrics.items()) + ', time: %.3f' % time)

    def set_num_iter(self, train_iter, test_iter):
        self.num_iter = {
            'train': train_iter,
            'test': test_iter}
