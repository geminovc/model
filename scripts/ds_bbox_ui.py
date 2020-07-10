import os
import numpy as np 
import cv2
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets  import RectangleSelector
import signal

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--folder', metavar='f', type=str,
                    help='path to the folder with all the images.')
parser.add_argument('--name', metavar='n', type=str,
                    help='name of the celebrity, used to create the pkl with the dictionary.')

args = parser.parse_args()

data_dir = args.folder
video_name = args.name
image_names = sorted(os.listdir(data_dir))

if not os.path.exists(video_name + ".pkl"):
    annotations = {}
    for x in image_names:
        annotations[x] = []

    with open(video_name + ".pkl", "wb") as handle:
        pickle.dump(annotations, handle)
else:
    with open(video_name + ".pkl", "rb") as handle:
        annotations = pickle.load(handle)
    assert(sorted(annotations.keys()) == image_names)

def line_select_callback(eclick, erelease):
    global prev_rect
    global current
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2), fill=False, edgecolor='red')
    if prev_rect != None:
        prev_rect.remove()

    prev_rect = rect
    current = [x1, y1, x2, y2]
    ax.add_patch(rect)

for i in range(len(image_names)):
    if len(annotations[image_names[i]]) != 0:
        print(image_names[i], "Done!")
        continue
    
    img = np.flip(cv2.imread(os.path.join(data_dir, image_names[i])), 2)
    xdata = np.linspace(0,9*np.pi, num=301)
    ydata = np.sin(xdata)

    fig, ax = plt.subplots()
    line = ax.imshow(img)

    prev_rect = None
    current = None


    rs = RectangleSelector(ax, line_select_callback,
                        drawtype='box', useblit=False, button=[1], 
                        minspanx=5, minspany=5, spancoords='pixels', 
                        interactive=True)
    plt.show()

    print(image_names[i], current)
    annotations[image_names[i]] = current
    with open(video_name + ".pkl", "wb") as handle:
        pickle.dump(annotations, handle)
