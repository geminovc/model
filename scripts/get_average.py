import numpy as np
import cv2
import sys

cap = cv2.VideoCapture(sys.argv[1])

# get video properties
width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

# compute the average of all frames
avg = np.zeros((height, width,3))
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    avg += np.true_divide(frame, float(total_frames))
cv2.imwrite(sys.argv[2], avg)

# When everything done, release the capture
cap.release()
