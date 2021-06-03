import torch
from torch import nn
import pdb
import argparse
import os
import pathlib
import importlib
import ssl
import time
import copy
import sys
import random 
from torch.utils.tensorboard import SummaryWriter
from datasets import utils as ds_utils
from networks import utils as nt_utils
from runners import utils as rn_utils
from logger import Logger



class TrainingWrapper(object):
    @staticmethod
    def get_args(parser):
        # General options
        parser.add('--experiment_dir',          default='.', type=str,
                                                help='directory to save logs')
        
        parser.add('--pretrained_weights_dir',  default='/video_conf/scratch/pantea', type=str,
                                                help='directory for pretrained weights of loss networks (lpips , ...)')
        
        parser.add('--project_dir',             default='.', type=str,
                                                help='root directory of the code')

        parser.add('--torch_home',              default='', type=str,
                                                help='directory used for storage of the checkpoints')

        parser.add('--experiment_name',         default='test', type=str,
                                                help='name of the experiment used for logging')

        parser.add('--dataloader_name',         default='voxceleb2', type=str,
                                                help='name of the file in dataset directory which is used for data loading')

        parser.add('--dataset_name',            default='voxceleb2_512px', type=str,
                                                help='name of the dataset in the data root folder')

        parser.add('--metrics_root',            default=".", type=str,
                                                help='root directory of the metrics')
        
        parser.add('--data_root',               default=".", type=str,
                                                help='root directory of the data')
        
        parser.add('--general_data_root',       default="/video-conf/scratch/pantea/video_conf_datasets/general_dataset", type=str,
                                                help='root directory of the general dataset, used for varying the weight of general to personal dataset')

        parser.add('--debug',                   action='store_true',
                                                help='turn on the debug mode: fast epoch, useful for testing')

        parser.add('--runner_name',             default='default', type=str,
                                                help='class that wraps the models and performs training and inference steps')

        parser.add('--no_disk_write_ops',       action='store_true',
                                                help='avoid doing write operations to disk')

        parser.add('--redirect_print_to_file',  default='False', type=rn_utils.str2bool, choices=[True, False],
                                                help='redirect stdout and stderr to file')

        parser.add('--random_seed',             default=0, type=int,
                                                help='used for initialization of pytorch and numpy seeds')

        # Initialization options
        parser.add('--init_experiment_dir',     default='', type=str,
                                                help='directory of the experiment used for the initialization of the networks')

        parser.add('--init_networks',           default='', type=str,
                                                help='list of networks to intialize')

        parser.add('--init_which_epoch',        default='none', type=str,
                                                help='epoch to initialize from')

        parser.add('--which_epoch',             default='none', type=str,
                                                help='epoch to continue training from')

        # Distributed options
        parser.add('--num_gpus',                default=1, type=int,
                                                help='>1 enables DDP')

        # Training options
        parser.add('--num_epochs',              default=1, type=int,
                                                help='number of epochs for training')

        parser.add('--num_metrics_images',      default=9, type=int,
                                                help='number of pairs of images in your metrics dir')

        parser.add('--checkpoint_freq',         default=25, type=int,
                                                help='frequency of checkpoints creation in epochs')

        parser.add('--test_freq',               default=5, type=int, 
                                                help='frequency of testing in epochs')
        
        parser.add('--metrics_freq',            default=5, type=int, 
                                                help='frequency of metrics in epochs')
        
        parser.add('--batch_size',              default=1, type=int,
                                                help='batch size across all GPUs')
        
        parser.add('--num_workers_per_process', default=20, type=int,
                                                help='number of workers used for data loading in each process')
        
        parser.add('--skip_test',               default='False', type=rn_utils.str2bool, choices=[True, False],
                                                help='do not perform testing')
        
        parser.add('--skip_metrics',            default='False', type=rn_utils.str2bool, choices=[True, False],
                                                help='do not perform metrics assessment')
        
        parser.add('--calc_stats',              action='store_true',
                                                help='calculate batch norm standing stats')
        
        parser.add('--visual_freq',             default=-1, type=int, 
                                                help='in iterations, -1 -- output logs every epoch')

        # Mixed precision options
        parser.add('--use_half',                action='store_true',
                                                help='enable half precision calculation')
        
        parser.add('--use_closure',             action='store_true',
                                                help='use closure function during optimization (required by LBFGS)')
        
        parser.add('--use_apex',                action='store_true',
                                                help='enable apex')
        
        parser.add('--amp_opt_level',           default='O0', type=str,
                                                help='full/mixed/half precision, refer to apex.amp docs')
        
        parser.add('--amp_loss_scale',          default='dynamic', type=str,
                                                help='fixed or dynamic loss scale')
        
        parser.add('--folder_postfix',          default='2d_crop', type=str,
                                                help='crop the stickman')
        
        parser.add('--output_segmentation',   default='False', type=rn_utils.str2bool, choices=[True, False],
                                              help='read segmentation mask')

        parser.add('--output_segmentation',     default='False', type=rn_utils.str2bool, choices=[True, False],
                                                help='read segmentation mask')
        
        parser.add('--metrics',                 default='PSNR, lpips, pose_matching', type=str,
                                                help='metrics to evaluate the model while training') 

        parser.add('--psnr_loss_apply_to',      default='pred_target_delta_lf_rgbs , target_imgs', type=str,
                                                help='psnr loss to apply') 
                                                                                              
        parser.add('--images_log_rate',         default=100, type=int,
                                                help='logging rate for images') 
        
        parser.add('--metrics_log_rate',        default=2, type=int,
                                                help='logging rate for metrics like PSNR') 
        
        parser.add('--nme_num_threads',         default=1, type=int,
                                                help='logging rate for images')     
                
        parser.add('--frame_num_from_paper',    default='False', type=rn_utils.str2bool, choices=[True, False],
                                                help='The random method to sample frame numbers for source and target from dataset')
        
        parser.add('--dataset_load_from_txt',   default='False', type=rn_utils.str2bool, choices=[True, False],
                                                help='If True, the train is loaded from train_load_from_filename, the test is loaded from test_load_from_filename. If false, the data is loaded from data-root')
        
        parser.add('--save_dataset_filenames',  default='False', type=rn_utils.str2bool, choices=[True, False],
                                                help='If True, the train/test data is saved in train/test_filnames.txt')
        
        parser.add('--train_load_from_filename',default='train_filnames.txt', type=str,
                                                help='filename that we read the training dataset images from if dataset_load_from_txt==True')                                    

        parser.add('--test_load_from_filename', default='test_filnames.txt', type=str,
                                                help='filename that we read the testing dataset images from if dataset_load_from_txt==True')  
        
        parser.add('--frozen_networks',         default='', type=str,
                                                help='list of frozen networks')

        parser.add('--networks_to_train',       default='identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator' , type=str,
                                                help='networks that are being trained')

        parser.add('--replace_Gtex_output_with_trainable_tensor',   default='False', type=rn_utils.str2bool, choices=[True, False],
                                                                    help='set to true if you want to replace all of G_tex with a tensor')

        parser.add('--unfreeze_texture_generator_last_layers',      default='True', type=rn_utils.str2bool, choices=[True, False],
                                                                    help='set to false if you want to freeze the last layers (after up samlping blocks) in the texture generator')

        parser.add('--unfreeze_inference_generator_last_layers',    default='False', type=rn_utils.str2bool, choices=[True, False],
                                                                    help='set to false if you want to freeze the last layers (after up samlping blocks) in the inference generator')

        parser.add('--save_initial_test_before_training',           default='True', type=rn_utils.str2bool, choices=[True, False],
                                                                    help='save how he model performs on test before training, useful for sanity check')

        parser.add('--augmentation_by_general',                     default='False', type=rn_utils.str2bool, choices=[True, False],
                                                                    help='gradually increase the weight of general dataset while training the per_person dataset')

        parser.add('--replace_source_specific_with_trainable_tensors',  default='False', type=rn_utils.str2bool, choices=[True, False],
                                                                        help='set to true if you want to replace all source-specific modules with a tensor')

        parser.add('--sample_general_dataset',                      default='False', type=rn_utils.str2bool, choices=[True, False],
                                                                        help='set to true if you want to take smaller number of data in general dataset')


                             


        # Technical options that are set automatically
        parser.add('--local_rank', default=0, type=int)
        parser.add('--rank',       default=0, type=int)
        parser.add('--world_size', default=1, type=int)
        parser.add('--train_size', default=1, type=int)

        # Dataset options
        args, _ = parser.parse_known_args()

        os.environ['TORCH_HOME'] = args.torch_home

        importlib.import_module(f'datasets.{args.dataloader_name}').DatasetWrapper.get_args(parser)

        # runner options
        importlib.import_module(f'runners.{args.runner_name}').RunnerWrapper.get_args(parser)
        

        return parser

    def __init__(self, args, runner=None):
        super(TrainingWrapper, self).__init__()
        # Initialize and apply general options
        ssl._create_default_https_context = ssl._create_unverified_context
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.random_seed)
        random.seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

        # Set distributed training options
        if args.num_gpus > 1 and args.num_gpus <= 8:
            args.rank = args.local_rank
            args.world_size = args.num_gpus
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')

        elif args.num_gpus > 8:
            raise # Not supported

        # Prepare experiment directories and save options
        experiment_dir = pathlib.Path(args.experiment_dir)
        
        self.checkpoints_dir = experiment_dir / 'runs' / args.experiment_name / 'checkpoints'

        # Store options
        if not args.no_disk_write_ops:
            os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.experiment_dir = experiment_dir / 'runs' / args.experiment_name

        if not args.no_disk_write_ops:
            # Redirect stdout
            if args.redirect_print_to_file:
                logs_dir = self.experiment_dir / 'logs'
                os.makedirs(logs_dir, exist_ok=True)
                sys.stdout = open(os.path.join(logs_dir, f'stdout_{args.rank}.txt'), 'w')
                sys.stderr = open(os.path.join(logs_dir, f'stderr_{args.rank}.txt'), 'w')

            if args.rank == 0:
                with open(self.experiment_dir / 'args.txt', 'wt') as args_file:
                    for k, v in sorted(vars(args).items()):
                        args_file.write('%s: %s\n' % (str(k), str(v)))

        # Initialize model
        self.runner = runner


        if self.runner is None:
            self.runner = importlib.import_module(f'runners.{args.runner_name}').RunnerWrapper(args)


        # Load pre-trained weights (if needed)
        init_networks = rn_utils.parse_str_to_list(args.init_networks) if args.init_networks else {}
        frozen_networks = rn_utils.parse_str_to_list(args.frozen_networks) if args.frozen_networks else {}
        networks_to_train = self.runner.nets_names_to_train
        #nets_frozen_networks_dict= rn_utils.parse_str_to_dict(args.frozen_networks_dict)

        if args.init_which_epoch != 'none' and args.init_experiment_dir:
            for net_name in init_networks:
                self.runner.nets[net_name].load_state_dict(torch.load(pathlib.Path(args.init_experiment_dir) / 'checkpoints' / f'{args.init_which_epoch}_{net_name}.pth', map_location='cpu'))    
                if net_name in frozen_networks: #dictionary
                    for p in self.runner.nets[net_name].parameters():
                        p.requires_grad = False
                    
                    #for unfreezed_layer_name in frozen_networks[net_name]['unfreezed_layers']:
                    #    getattr(self.runner.nets[net_name], unfreezed_layer_name).requires_grad=True
                    #    --frozen_networks '{'texture_generator': {'unfreezed_layers':['layer_1','layer_2']}}'
                    #frozen_networks ={'texture_generator': {'unfreezed_layers':['layer_1','layer_2']}}
                if net_name == "texture_generator" and net_name in frozen_networks and args.unfreeze_texture_generator_last_layers: 
                    for name, module in self.runner.nets[net_name].named_children():
                        if name == 'prj_tex':
                            print("unfreezing prj_tex for texture generator in gen_tex ...")
                            for p in module.parameters():
                                p.requires_grad = True
                        if name =='gen_tex':
                            for subname, submodule in module.named_children():
                                if subname == 'heads':
                                    print("unfreezing heads for texture generator in gen_tex ...")
                                    for p in submodule.parameters():
                                        p.requires_grad = True
                                if subname == 'blocks':
                                    for subsubname, subsubmodule in submodule.named_children():
                                        if int(subsubname) > 5:
                                            print("unfreezing after AdaSpade for texture generator in gen_tex ...")
                                            for p in subsubmodule.parameters():
                                                p.requires_grad = True



                if net_name == "inference_generator" and net_name in frozen_networks and args.unfreeze_inference_generator_last_layers: 
                    for name, module in self.runner.nets[net_name].named_children():
                        if name =='prj_inf':
                            print("unfreezing prj_inf for inference generator in gen_inf ...")
                            for p in module.parameters():
                                p.requires_grad = True                  
                        if name =='gen_inf':
                            for subname, submodule in module.named_children():
                                if subname == 'heads':
                                    print("unfreezing heads for inference generator in gen_inf ...")
                                    for p in submodule.parameters():
                                        p.requires_grad = True
                                if subname == 'blocks':
                                    for subsubname, subsubmodule in submodule.named_children():
                                        if int(subsubname) > 5:
                                            print("unfreezing after AdaSpade for inference generator in gen_inf ...")
                                            for p in subsubmodule.parameters():
                                                p.requires_grad = True




        if args.which_epoch != 'none':
            for net_name in networks_to_train:
                if net_name not in init_networks:
                    self.runner.nets[net_name].load_state_dict(torch.load(self.checkpoints_dir / f'{args.which_epoch}_{net_name}.pth', map_location='cpu'))

                    
        if args.num_gpus > 0:
            self.runner.cuda()

        if args.rank == 0:
            print()

        # If we are reading from the data filenames from a txt file, there is no need to store it again
        # commented to test 
        # if args.dataset_load_from_txt:
        #     args.save_dataset_filenames = False

        if args.save_dataset_filenames:
            print("Clearing the files already stored as train_filenames.txt and test_filenames.txt.")
            train_file = "train_filenames.txt"
            file = open(self.experiment_dir / train_file,"w+")
            file.truncate(0)
            file.write('data-root:%s\n' % (str(args.data_root)))
            file.close()
            # with open(self.experiment_dir / train_file, 'a') as data_file:
            #     data_file.write('\n')


            test_file = "test_filenames.txt"
            file = open(self.experiment_dir / test_file,"w+")
            file.truncate(0)
            file.write('data-root:%s\n' % (str(args.data_root)))
            file.close()
            # with open(self.experiment_dir / test_file, 'a') as data_file:
            #     data_file.write('\n')


    def get_current_lr(self, optimizer, group_idx, parameter_idx, step):
        # Adam has different learning rates for each paramter. So we need to pick the
        # group and paramter first.
        group = optimizer.param_groups[group_idx]
        p = group['params'][parameter_idx]

        beta1, _ = group['betas']
        state = optimizer.state[p]

        bias_correction1 = 1 - beta1 ** step
        current_lr = group['lr'] / bias_correction1
        return current_lr


    def train(self, args):
        # Reset amp
        if args.use_apex:
            from apex import amp
            
            amp.init(False)

        # Tensorboard writer init
        if args.rank == 0:
            writer = SummaryWriter(log_dir= args.experiment_dir + '/runs/' + args.experiment_name + '/metrics/')

        # Get relevant dataloaders for augmentation by general or the vanilla case
        if args.augmentation_by_general and args.data_root != args.general_data_root:
            print("getting the per_person dataset")
            personal_train_dataloader = ds_utils.get_dataloader(args, 'train')
            self.gen_to_per_ratio = 1
            self.real_data_root = args.data_root
            args.data_root = args.general_data_root
            print("getting the general dataset")
            general_train_dataloader = ds_utils.get_dataloader(args, 'train')
            args.data_root = self.real_data_root
        else:
            original_train_dataloader = ds_utils.get_dataloader(args, 'train')

        if not args.skip_test:
            test_dataloader = ds_utils.get_dataloader(args, 'test')
        if not args.skip_metrics:
            metrics_dataloader = ds_utils.get_dataloader(args, 'metrics')




        model = runner = self.runner

        if args.use_half:
            runner.half()

        # Initialize optimizers, schedulers and apex
        opts = runner.get_optimizers(args)

        # Load pre-trained params for optimizers and schedulers (if needed)
        if args.which_epoch != 'none' and not args.init_experiment_dir:
            for net_name, opt in opts.items():
                opt.load_state_dict(torch.load(self.checkpoints_dir / f'{args.which_epoch}_opt_{net_name}.pth', map_location='cpu'))

        if args.use_apex and args.num_gpus > 0 and args.num_gpus <= 8:
            # Enfornce apex mixed precision settings
            nets_list, opts_list = [], []
            for net_name in sorted(opts.keys()):
                nets_list.append(runner.nets[net_name])
                opts_list.append(opts[net_name])

            loss_scale = float(args.amp_loss_scale) if args.amp_loss_scale != 'dynamic' else args.amp_loss_scale

            nets_list, opts_list = amp.initialize(nets_list, opts_list, opt_level=args.amp_opt_level, num_losses=1, loss_scale=loss_scale)

            # Unpack opts_list into optimizers
            for net_name, net, opt in zip(sorted(opts.keys()), nets_list, opts_list):
                runner.nets[net_name] = net
                opts[net_name] = opt

            if args.which_epoch != 'none' and not args.init_experiment_dir and os.path.exists(self.checkpoints_dir / f'{args.which_epoch}_amp.pth'):
                amp.load_state_dict(torch.load(self.checkpoints_dir / f'{args.which_epoch}_amp.pth', map_location='cpu'))

        # Initialize apex distributed data parallel wrapper
        if args.num_gpus > 1 and args.num_gpus <= 8:
            from apex import parallel

            model = parallel.DistributedDataParallel(runner, delay_allreduce=True)

        epoch_start = 1 if args.which_epoch == 'none' else int(args.which_epoch) + 1

        # Initialize logging
        train_iter = epoch_start - 1

        if args.visual_freq != -1:
            train_iter /= args.visual_freq

        logger = Logger(args, self.experiment_dir)
        logger.set_num_iter(
            train_iter=train_iter, 
            test_iter=(epoch_start - 1) // args.test_freq,
            metrics_iter=(epoch_start - 1) // args.metrics_freq)

        if args.debug and not args.use_apex:
            torch.autograd.set_detect_anomaly(True)

        total_iters = 1
        iter_count = 0

        # Adding the first test image on the logger for sanity check
        if args.save_initial_test_before_training:
            print("Testing the model before starts training for sanity check")
            if args.augmentation_by_general and args.data_root!=args.general_data_root:
                train_dataloader = personal_train_dataloader
            else:
                train_dataloader = original_train_dataloader
            # Calculate "standing" stats for the batch normalization
            train_dataloader.dataset.shuffle()
            if args.calc_stats:
                runner.calculate_batchnorm_stats(train_dataloader, args.debug)

            # Test
            time_start = time.time()
            model.eval()

            test_dataloader.dataset.shuffle()
            for data_dict in test_dataloader:
                # Prepare input data
                if args.num_gpus > 0:
                    for key, value in data_dict.items():
                        data_dict[key] = value.cuda()
                # Forward pass
                with torch.no_grad():
                    model(data_dict)
                
                if args.debug:
                    break

            # Output logs
            logger.output_logs('test', runner.output_visuals(), runner.output_losses(), time.time() - time_start)
            




        for epoch in range(epoch_start, args.num_epochs + 1):
            self.epoch_start = time.time()
            if args.rank == 0: 
                print('epoch %d' % epoch)

            # Train for one epoch from now on

            # Initiate all the networks in the training mode 
            model.train() 
            time_start = time.time()
            if args.augmentation_by_general and args.data_root!=args.general_data_root:
                prob = random.uniform(0, 1)
                self.gen_to_per_ratio = (args.num_epochs-epoch)/(args.num_epochs-epoch_start)
                self.test_freq = 5
                if prob < self.gen_to_per_ratio:
                    print("selecting from the general dataset ...")
                    train_dataloader = general_train_dataloader
                else:
                    print("selecting from the per_person dataset ...")
                    train_dataloader = personal_train_dataloader
            else:
                train_dataloader = original_train_dataloader
                
            
            # Shuffle the dataset before the epoch
            train_dataloader.dataset.shuffle()
            for i, data_dict in enumerate(train_dataloader, 1):
                iter_count+=1 
                # Prepare input data
                if args.num_gpus > 0 and args.num_gpus > 0:
                    for key, value in data_dict.items():
                        data_dict[key] = value.cuda()

                # Convert inputs to FP16
                if args.use_half:
                    for key, value in data_dict.items():
                        data_dict[key] = value.half()

                output_logs = i == len(train_dataloader)

                if args.visual_freq != -1:
                    output_logs = not (total_iters % args.visual_freq)

                output_visuals = output_logs and not args.no_disk_write_ops

                # Accumulate list of optimizers that will perform opt step
                for opt in opts.values():
                    opt.zero_grad()

                # Perform a forward pass
                if not args.use_closure:
                    loss = model(data_dict)
                    closure = None

                if args.use_apex and args.num_gpus > 0 and args.num_gpus <= 8:
                    # Mixed precision requires a special wrapper for the loss
                    with amp.scale_loss(loss, opts.values()) as scaled_loss:
                        scaled_loss.backward()

                elif not args.use_closure:
                    loss.backward()

                else:
                    def closure():
                        loss = model(data_dict)
                        loss.backward()
                        return loss

                # Perform steps for all optimizers
                for opt in opts.values():
                    opt.step(closure)

                print("The time for this epoch is:", time.time() - time_start)
                if output_logs:
                    logger.output_logs('train', runner.output_visuals(), runner.output_losses(), \
                            runner.output_metrics(), time.time() - time_start)
 
                    if args.debug:
                        break

                if args.visual_freq != -1:
                    total_iters += 1
                    total_iters %= args.visual_freq
            
            # Increment the epoch counter in the training dataset
            train_dataloader.dataset.epoch += 1

            # If skip test flag is set -- only check if a checkpoint if required
            if not args.skip_test and not epoch % args.test_freq:
                # Calculate "standing" stats for the batch normalization
                if args.calc_stats:
                    runner.calculate_batchnorm_stats(train_dataloader, args.debug)

                # Test
                time_start = time.time()
                model.eval()

                test_dataloader.dataset.shuffle()
                for data_dict in test_dataloader:
                    # Prepare input data
                    if args.num_gpus > 0:
                        for key, value in data_dict.items():
                            data_dict[key] = value.cuda()

                    

                    # Forward pass
                    with torch.no_grad():
                        model(data_dict)
                    
                    if args.debug:
                        break
                logger.output_logs('test', runner.output_visuals(), runner.output_losses(), \
                    runner.output_metrics(), time.time() - time_start)

            # If skip test flag is set -- only check if a checkpoint if required
            if not args.skip_metrics and not epoch % args.metrics_freq:
                # Calculate "standing" stats for the batch normalization
                if args.calc_stats:
                    runner.calculate_batchnorm_stats(train_dataloader, args.debug)

                # Test
                time_start = time.time()
                model.eval()

                for i, data_dict in enumerate(metrics_dataloader, 1):
                    # Prepare input data
                    if args.num_gpus > 0:
                        for key, value in data_dict.items():
                            data_dict[key] = value.cuda()

                    # Forward pass
                    with torch.no_grad():
                        model(data_dict)
                    
                    if args.debug:
                        break

                    logger.output_logs('metrics', runner.output_visuals(), runner.output_losses(), \
                            runner.output_metrics(), time.time() - time_start, i)

            # If creation of checkpoint is not required -- continue
            if epoch % args.checkpoint_freq and not args.debug:
                continue
            # Create or load a checkpoint
            if args.rank == 0  and not args.no_disk_write_ops:
                with torch.no_grad():
                    for net_name in runner.nets_names_to_train:
                        # Save a network
                        torch.save(runner.nets[net_name].state_dict(), self.checkpoints_dir / f'{epoch}_{net_name}.pth')

                        # Save an optimizer
                        torch.save(opts[net_name].state_dict(), self.checkpoints_dir / f'{epoch}_opt_{net_name}.pth')

                    # Save amp
                    if args.use_apex:
                        torch.save(amp.state_dict(), self.checkpoints_dir / f'{epoch}_amp.pth')
        print("The epoch", str(epoch),  "took (s):", time.time()-self.epoch_start)
        
        return runner

if __name__ == "__main__":
    ## Parse options ##
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add = parser.add_argument

    TrainingWrapper.get_args(parser)

    args, _ = parser.parse_known_args()

    ## Initialize the model ##
    m = TrainingWrapper(args)

    ## Perform training ##
    nets = m.train(args)
