import pickle
import os 
import csv

# open a file, where you stored the pickled dictionary
file = open('/video-conf/scratch/pantea_experiments_chunky/runs/metrics_new_keypoints_G_inf_and_last_G_tex_unfrozen/metrics2.pkl', 'rb')

# dump information to that file
my_dict = pickle.load(file)

# close the file
file.close()

for key in my_dict:
    print(key)
    print(my_dict[key])


with open('metrics.csv', 'w') as f:
    for key in my_dict.keys():
        f.write("%s,%s\n"%(key,my_dict[key]))