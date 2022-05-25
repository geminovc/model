import os
import argparse
import glob
import numpy as np
from PIL import Image
import math


# arguments
parser = argparse.ArgumentParser('Video metric computer')
parser.add_argument('--result-file',
                    type=str,
                    help='where to write results to')
parser.add_argument('--base-dir',
                    type=str,
                    help='base folder location', 
                    default="/video-conf/scratch/vibhaa_mm_log_directory/")
parser.add_argument('--metric-list',
                    type=str,
                    nargs='+',
                    default=["SSIM", "LPIPS", "PSNR"],
                    help='what metrics to parse')
parser.add_argument('--video-prefix',
                    type=str,
                    default='conan',
                    help='session name to aggregate metrics over') 
parser.add_argument('--experiment-log-list',
                    type=str,
                    nargs='+',
                    help='list of logs to combine')
parser.add_argument('--experiment-name-list',
                    type=str,
                    nargs='+',
                    help='names of experiments')
args = parser.parse_args()

result_file = open(args.result_file, "w+")
result_file.write("experiment,metric,value\n")
experiment_log_list = args.experiment_log_list
experiment_name_list = args.experiment_name_list

for folder, exp_name in zip(experiment_log_list, experiment_name_list):
    full_folder = os.path.join(args.base_dir, folder)
    summary_file_list = glob.glob(full_folder + "/*metrics_summary.txt")

    metrics = {m: [] for m in args.metric_list}
    for summary in summary_file_list:
        with open(summary, "r") as f:
            for line in f.readlines():
                words = line.split(" ")
                video_name = words[0]
                if args.video_prefix in video_name:
                    metrics_str = "".join(words[1:])[:-1]
                    for m in metrics_str.split(","):
                        name = m.split(":")[0]
                        value = m.split(":")[1]
                        if name in args.metric_list:
                            metrics[name].append(float(value))

    for m in metrics:
        result_file.write(f"{exp_name},{m},{np.average(metrics[m])}\n")

result_file.close()
