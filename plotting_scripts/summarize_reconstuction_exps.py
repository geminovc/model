
"""
This script loads the saved pickle files from multiple experiments and saves them in csv format.
Sample usage:
python summarize_reconstuction_exps.py --result-file-list pkl_path_1 pkl_path_2 
--experiment-name-list scheme_1 scheme_2 --result-file-name metrics.csv
"""

#Importing libraries
import pickle
import os 
import csv
import pdb
import argparse

parser= argparse.ArgumentParser("Data Summarizer")
parser.add_argument('--result-file-list',
        type=str,
        nargs='+',
        help='list of result files to aggregate data into csv for')
parser.add_argument('--experiment-name-list',
        type=str,
        nargs='+',
        help='associated names for diff experiments when aggregating into csv')
parser.add_argument('--result-file-name',
        type=str,
        required=True,
        help='name of csv file to write results out to')
parser.add_argument('--window',
        type=int,
        default=10,
        help='number of epochs to compute metrics over')
args = parser.parse_args()
window = args.window


# This function receives a path string and outputs the .pkl file associated with it
def load_pickle(path_string):
    pkl_file = open(path_string, 'rb')
    my_dict = pickle.load(pkl_file)
    pkl_file.close()
    return my_dict

# This function averages over the last window_size of input dict
def average (my_dict, key, window_size):
    temp = my_dict[key]
    float_values = [x.item() for x in temp]
    float_values = float_values[-window_size:]
    return sum(float_values) / len(float_values)
    
# This function finds the maximum over the last window_size of input dict
def max_in_window (my_dict, key, window_size):
    temp = my_dict[key]
    float_values = [x.item() for x in temp]
    float_values = float_values[-window_size:]
    return max(float_values) 
    
# This function finds the minimum over the last window_size of input dict
def min_in_window (my_dict, key, window_size):
    temp = my_dict[key]
    float_values = [x.item() for x in temp]
    float_values = float_values[-window_size:]
    return min(float_values) 

# Load the pickle files one at a time and write to csv
with open(args.result_file_name, 'w') as f:
    f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%('experiment_name','G_PSNR_min', 'G_PSNR_mean', 
            'G_PSNR_max', 'G_LPIPS_min','G_LPIPS_mean','G_LPIPS_max', 'G_PME_min', 'G_PME_mean','G_PME_max',
            'G_CSIM_min','G_CSIM_mean','G_CSIM_max','G_SSIM_min','G_SSIM_mean','G_SSIM_max'))
    
    for experiment_name, result_file in zip(args.experiment_name_list, args.result_file_list):
        experiment_data = load_pickle(result_file)
        f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (experiment_name, \
                                                  min_in_window (experiment_data, 'G_PSNR', window),
                                                  average (experiment_data, 'G_PSNR', window),
                                                  max_in_window (experiment_data, 'G_PSNR', window),
                                                  min_in_window (experiment_data, 'G_LPIPS', window),
                                                  average (experiment_data, 'G_LPIPS', window),
                                                  max_in_window (experiment_data, 'G_LPIPS', window),
                                                  min_in_window (experiment_data, 'G_PME', window),
                                                  average (experiment_data, 'G_PME', window),
                                                  max_in_window (experiment_data, 'G_PME', window),
                                                  min_in_window (experiment_data, 'G_CSIM', window),
                                                  average (experiment_data, 'G_CSIM', window),
                                                  max_in_window (experiment_data, 'G_CSIM', window),
                                                  min_in_window (experiment_data, 'G_SSIM', window),
                                                  average (experiment_data, 'G_SSIM', window),
                                                  max_in_window (experiment_data, 'G_SSIM', window),
                                                  ))

print('Saving to file Successfully done!')