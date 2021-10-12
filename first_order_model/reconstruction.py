import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
from sync_batchnorm import DataParallelWithCallback

""" helper to get size of nested parameter list """
def get_size_of_nested_list(list_of_elem):
    count = 0
    for elem in list_of_elem:
        if type(elem) == list:  
            count += get_size_of_nested_list(elem)
        else:
            count += 1    
    return count

""" get model summary information for the passed in keypoint detector and 
    generator in a text file in the log directory 
"""
def get_model_info(log_dir, kp_detector, generator):
    with open(os.path.join(log_dir, 'model_summary.txt'), 'wt') as model_file:
        for model, name in zip([kp_detector, generator], ['kp', 'generator']):
            number_of_trainable_parameters = 0
            total_number_of_parameters = 0
            for param in model.parameters():
                total_number_of_parameters += get_size_of_nested_list(list(param))
                if param.requires_grad:
                    number_of_trainable_parameters += get_size_of_nested_list(list(param))

            model_file.write('%s %s: %s\n' % (name, 'total_number_of_parameters', \
                    str(total_number_of_parameters)))
            model_file.write('%s %s: %s\n' % (name, 'number_of_trainable_parameters', \
                    str(number_of_trainable_parameters)))


""" reconstruct driving frames for each video in the dataset using the first frame
    as a source frame. Config specifies configration details, while timing 
    determines whether to time the functions on a gpu or not
"""
def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset, timing_enabled):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    loss_list = []
    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    # get number of model parameters and timing stats
    get_model_info(log_dir, kp_detector, generator)
    start = torch.cuda.Event(enable_timing=timing_enabled)
    end = torch.cuda.Event(enable_timing=timing_enabled)

    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            
            start.record()
            kp_source = kp_detector(x['video'][:, :, 0])
            end.record()
            torch.cuda.synchronize()
            
            if timing_enabled:
                source_time = start.elapsed_time(end)
                driving_times, generator_times, visualization_times = [], [], []

            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]
                driving = x['video'][:, :, frame_idx]
                
                start.record()
                kp_driving = kp_detector(driving)
                end.record()
                torch.cuda.synchronize()
                if timing_enabled:
                    driving_times.append(start.elapsed_time(end))
                
                start.record()
                out = generator(source, kp_source=kp_source, kp_driving=kp_driving)
                end.record()
                torch.cuda.synchronize()
                if timing_enabled:
                    generator_times.append(start.elapsed_time(end))
                
                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                del out['sparse_deformed']

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                start.record()
                visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
                                                                                    driving=driving, out=out)
                end.record()
                torch.cuda.synchronize()
                if timing_enabled:
                    visualization_times.append(start.elapsed_time(end))
                visualizations.append(visualization)

                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())

            predictions = np.concatenate(predictions, axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)
            
            if timing_enabled:
                print("source keypoints:", source_time, "driving:", np.average(driving_times), \
                    "generator:", np.average(generator_times), "visualization:", np.average(visualization_times))

    print("Reconstruction loss: %s" % np.mean(loss_list))
