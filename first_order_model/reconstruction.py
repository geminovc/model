import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
from sync_batchnorm import DataParallelWithCallback
from skimage.metrics import peak_signal_noise_ratio
from skimage import img_as_float32
from skimage.metrics import structural_similarity
from frames_dataset import get_num_frames, get_frame
import lpips

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

""" get average of visual metrics across all frames
"""
def get_avg_visual_metrics(visual_metrics):
    psnrs = [m['psnr'] for m in visual_metrics]
    ssims = [m['ssim'] for m in visual_metrics]
    lpips_list = [m['lpips'] for m in visual_metrics]
    return np.mean(psnrs), np.mean(ssims), np.mean(lpips_list)

"""convert numpy arrays to tensors for reconstruction pipeline
"""
def frame_to_tensor(frame, device):
    array = np.expand_dims(frame, 0).transpose(0, 3, 1, 2)
    array = torch.from_numpy(array)
    return array.float().to(device)

""" reconstruct driving frames for each video in the dataset using the first frame
    as a source frame. Config specifies configration details, while timing 
    determines whether to time the functions on a gpu or not
"""
def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset, timing_enabled, 
        save_visualizations_as_images, experiment_name, reference_frame_update_freq=None):
    log_dir = os.path.join(log_dir, 'reconstruction' + '_' + experiment_name)
    png_dir = os.path.join(log_dir, 'png')
    visualization_dir = os.path.join(log_dir, 'visualization')
    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
    
    metrics_file = open(os.path.join(log_dir, experiment_name + '_metrics_summary.txt'), 'wt')
    loss_list = []
    visual_metrics = []
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        loss_fn_vgg = loss_fn_vgg.cuda()
 
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
            video_name = x['video_path'][0]
            reader = imageio.get_reader(video_name, "ffmpeg")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if timing_enabled:
                source_time = start.elapsed_time(end)
                driving_times, generator_times, visualization_times = [], [], []

            frame_idx = 0
            for frame in reader:
                frame = img_as_float32(frame)
                if frame_idx == 0:
                    source = frame_to_tensor(frame, device)
                    start.record()
                    kp_source = kp_detector(source)
                    end.record()
                    torch.cuda.synchronize()

                if reference_frame_update_freq is not None:
                    if frame_idx % reference_frame_update_freq == 0:
                        source = frame_to_tensor(frame, device) 
                        kp_source = kp_detector(source)
                
                driving = frame_to_tensor(frame, device)
                frame_idx += 1
                
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

                start.record()
                visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
                                                                                    driving=driving, out=out)
                end.record()
                torch.cuda.synchronize()
                if timing_enabled:
                    visualization_times.append(start.elapsed_time(end))
                visualizations.append(visualization)

                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())
                visual_metrics.append(Logger.get_visual_metrics(out['prediction'], driving, loss_fn_vgg))
                
                last_prediction = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

            predictions = np.concatenate(predictions, axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), 
                    (255 * predictions).astype(np.uint8))

            image_name = x['name'][0] + config['reconstruction_params']['format']

            psnr, ssim, lpips_val = get_avg_visual_metrics(visual_metrics)
            metrics_file.write("%s PSNR: %s, SSIM: %s, LPIPS: %s\n" % (x['name'][0], 
                    psnr, ssim, lpips_val))
            metrics_file.flush()

            if save_visualizations_as_images:
                for i, v in enumerate(visualizations):
                    frame_name = x['name'][0] + '_frame' + str(i) + '.png'
                    imageio.imsave(os.path.join(visualization_dir, frame_name), v)

            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)
            
            if timing_enabled:
                print("source keypoints:", source_time, "driving:", np.average(driving_times), \
                    "generator:", np.average(generator_times), "visualization:", np.average(visualization_times))

    print("Reconstruction loss: %s" % np.mean(loss_list))
    metrics_file.write("Reconstruction loss: %s\n" % np.mean(loss_list))
    metrics_file.close()

