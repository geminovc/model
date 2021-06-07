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
    

#dict_metrics = load_pickle('/video-conf/scratch/pantea_experiments_mapmaker/runs/metrics_new_keypoints_from_base/metrics_metrics.pkl')
#dict_test    = load_pickle('/video-conf/scratch/pantea_experiments_mapmaker/runs/metrics_new_keypoints_from_base/metrics_test.pkl')
dict_train    = load_pickle('/video-conf/scratch/pantea_experiments_mapmaker/runs/metrics_new_keypoints_G_inf_and_last_G_tex_unfrozen/metrics.pkl')
print(average (dict_train, 'G_PSNR', 10))

field_names = dict_train.keys()
# with open('metrics.csv', 'w') as f:
#     for key in my_dict.keys():
#         temp = my_dict[key]
#         float_values = [x.item() for x in temp]
#         print(float_values)
#         #f.write("%s,%s\n"%(key,my_dict[key].item()))

with open('metrics.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = ['G_PSNR', 'G_LPIPS', 'G_PME'])
    writer.writeheader()
    #writer.writerows(dict_train)
    for data in dict_train:
        writer.writerow(data)