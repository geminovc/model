import cv2
import os
import argparse
import glob
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import math

# transform for centering picture
regular_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# globals
DATASET_PATH= "/data/vibhaa/speech2gesture-master/dataset"
loss_fn = lpips.LPIPS(net='alex')

# arguments
parser = argparse.ArgumentParser('Video metric computer')
parser.add_argument('--result-file',
                type=str,
                help='where to write results to')
parser.add_argument('--metric',
                type=str,
                default='SSIM',
                help='where to write results to', choices=['SSIM', 'LPIPS', 'PSNR'])
parser.add_argument('--speaker',
                type=str,
                default='conan',
                help='speaker to focus on', 
                choices=['conan', 'angelica', 'almaram', 'chemistry', 'ellen', 'jon', 'oliver', 'rock', 'seth', 'shelly'])
parser.add_argument('--video-prefix',
                type=str,
                help='prefix of the input video to compute metrics for ')
parser.add_argument('--kilo-bitrate-list',
                type=int,
                nargs='+',
                help='list of kb values to aggregate results over', default=[])
args = parser.parse_args()

""" custom psnr calculator """
def per_frame_psnr(x, y):
    assert(x.size == y.size)

    mse = np.mean(np.square((x - y)/255.0))
    if mse > 0:
        psnr = -10 * math.log10(mse)
    else:
        psnr = 100000
    return psnr

""" compute metric for all filenames matching prefix relative to the reference video """
def compute_metric_for_files(video_prefix, setting):
    # read one frame at a time and compute average metric
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    num_frames = 0
    
    reference_list = glob.glob(DATASET_PATH + "/" + \
            args.speaker + "/spatially_cropped/" + video_prefix + "*")

    for ref_filename in reference_list:
        filename = os.path.basename(os.path.splitext(ref_filename)[0])
        input_filename = DATASET_PATH + "/" + args.speaker + "/downsampled/" + filename
        input_filename += "_" + setting + ".mp4"
        
        megabit_ref_filename = DATASET_PATH + "/" + \
                args.speaker + "/downsampled/" + filename + "_1000Kb.mp4"

        # set up reader
        cap_lowres = cv2.VideoCapture(input_filename)
        cap_reference = cv2.VideoCapture(megabit_ref_filename)
    
        # read frame
        ret_input, input_frame = cap_lowres.read()
        ret_ref, reference_frame = cap_reference.read()
        
        while ret_input and ret_ref:
            num_frames += 1
            
            ssim_sum += ssim(reference_frame, input_frame, multichannel=True)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                psnr_sum += psnr(reference_frame, input_frame)

            # resize input into square, crop the center 256x256 
            w, h, _ = input_frame.shape
            if w < h:
                input_frame = input_frame[:, 0:w, :]
                reference_frame = reference_frame[:, 0:w, :]
            else:
                input_frame = input_frame[0:h, :, :]
                reference_frame = reference_frame[0:h, :, :]
            input_frame = regular_transform(Image.fromarray(input_frame))
            reference_frame = regular_transform(Image.fromarray(reference_frame))
            
            lpips_sum += loss_fn(reference_frame.unsqueeze(0), input_frame.unsqueeze(0))
           
            ret_input, input_frame = cap_lowres.read()
            ret_ref, reference_frame = cap_reference.read()

        cap_lowres.release()
        cap_reference.release()

    lpips_sum = torch.flatten(lpips_sum.detach())[0].numpy()
    return np.array([psnr_sum, ssim_sum, lpips_sum]) / num_frames

def main():
    with open(args.result_file, "w+") as f:
        f.write("setting,psnr,ssim,lpips\n")

        for b in args.kilo_bitrate_list:
            metrics = compute_metric_for_files(args.video_prefix, str(b) + "Kb")
            print(metrics)
            f.write(str(b) + "Kb," + ",".join([str(m) for m in metrics]) + "\n")  

main()
