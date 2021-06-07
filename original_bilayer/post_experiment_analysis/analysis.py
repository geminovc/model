"""
This script loads the saved pickle files from multiple experiments and saves them in csv format.
"""

#Importing libraries
import pickle
import os 
import csv
import pdb 


# This function receives a path string and outputs the .pkl file associated with it
def load_pickle(path_string):
    # open a file, where you stored the pickled dictionary
    file = open(path_string, 'rb')
    # dump information to that file
    my_dict = pickle.load(file)
    # close the file
    file.close()
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

# Load the pickle files 
"""
the data_dict is in the format of [['experiment_0_name', 'experiment_0_result'], ..., ['experiment_n_name', 'experiment_n_result']]
window is the size of the average, min, and max window over which the output for each metric in experiments is computed.  
"""

data_dict = []
data_dict.append(['G_inf',load_pickle('/video-conf/scratch/pantea_experiments_mapmaker/runs/metrics_new_keypoints_G_inf_and_last_G_tex_unfrozen/metrics.pkl')])
data_dict.append(['No_frozen', load_pickle('/video-conf/scratch/pantea_experiments_mapmaker/runs/metrics_new_keypoints_no_frozen/metrics.pkl')])
window = 10


# Write the value in the csv format
with open('metrics.csv', 'w') as f:
    f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%('experiment_name','G_PSNR_min', 'G_PSNR_mean', 'G_PSNR_max', 'G_LPIPS_min','G_LPIPS_mean','G_LPIPS_max', 'G_PME_min', 'G_PME_mean','G_PME_max'))
    
    #Itrate over the experiments
    for i in range(0, len(data_dict)):
        experiment_name = data_dict[i][0]
        experiment_data = data_dict[i][1]
        f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(experiment_name, min_in_window (experiment_data, 'G_PSNR', window),
                                                  average (experiment_data, 'G_PSNR', window),
                                                  max_in_window (experiment_data, 'G_PSNR', window),
                                                  min_in_window (experiment_data, 'G_LPIPS', window),
                                                  average (experiment_data, 'G_LPIPS', window),
                                                  max_in_window (experiment_data, 'G_LPIPS', window),
                                                  min_in_window (experiment_data, 'G_PME', window),
                                                  average (experiment_data, 'G_PME', window),
                                                  max_in_window (experiment_data, 'G_PME', window),
                                                  ))

print('Saving to file Successfully done!')