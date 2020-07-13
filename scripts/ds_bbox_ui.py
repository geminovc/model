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

def press(event):
    global i
    global img
    global current
    global prev_rect
    global ax
    if event.key == 'z':
        if current != None:
            plt.close()

    if event.key == 'a':
        if current == None:
            print("Please create an annotation before adjusting it.")
        prev_rect.remove()
        x1, y1, x2, y2 = current
        center_x = np.abs(x1+x2)/2
        center_y = np.abs(y1+y2)/2
        length = min(img.shape[0], img.shape[1])
        start_x = int(center_x - length/2)
        start_y = int(center_y - length/2)
        rect = plt.Rectangle((start_x, start_y), int(length), int(length), fill=False, edgecolor='blue')
        prev_rect = rect
        current = [max(0, start_x), max(0, start_y), min(img.shape[0], start_x + length), min(img.shape[1],start_y + length)]
        ax.add_patch(rect)
        plt.draw()
        plt.show()

prev_rect = None
current = []
prev_dim = None

for i in range(len(image_names)):
    if len(annotations[image_names[i]]) != 0:
        print(image_names[i], "Done!")
        current = annotations[image_names[i]]
        continue

    img = np.flip(cv2.imread(os.path.join(data_dir, image_names[i])), 2)

    xdata = np.linspace(0,9*np.pi, num=301)
    ydata = np.sin(xdata)

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', press)
    line = ax.imshow(img)        
    
    if i != 0 and img.shape != prev_dim:
        if prev_rect != None:
            prev_rect.remove()
        
        prev_rect = None
        current = []
    
    print(current)
    if current != None and len(current) != 0:
        x1, y1, x2, y2 = current
        cur_rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2), fill=False, edgecolor='red')
        ax.add_patch(cur_rect)
        prev_rect = cur_rect

    rs = RectangleSelector(ax, line_select_callback,
                        drawtype='box', useblit=False, button=[1], 
                        minspanx=5, minspany=5, spancoords='pixels',
                        interactive=True)
    plt.show()
    prev_dim = img.shape
    print(image_names[i], current)
    annotations[image_names[i]] = current
    with open(video_name + ".pkl", "wb") as handle:
        pickle.dump(annotations, handle)
