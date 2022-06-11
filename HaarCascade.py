#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMPLEMENTATION OF HAAR CASCADE CLASSIFIER METHOD
    1) From the terminal, input "python HaarCascade.py <videoFile>"
    2) An outputted video file tracks the bottommost changes in the video.
    3) An outputted text file stores the centers of the bottommost changes.
"""
import numpy as np
import cv2, os, sys, time

initTime = time.time()

# LOAD IN VIDEO
videoFile = sys.argv[1]
videoIn = cv2.VideoCapture(videoFile)

# GET DIMENSIONS
W = int(videoIn.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(videoIn.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = videoIn.get(cv2.CAP_PROP_FPS)
xvals = []
yvals = []
wvals = []
hvals = []
xmax = []
ymax = []

# INITIALIZE
count = 0
jumpFrames = 3
low_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lowerbody.xml')

# OUTPUT VIDEO
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
videoOut = cv2.VideoWriter("haar_" + videoFile, fourcc, fps/jumpFrames, (int(H),int(W)))

# BEGIN READING
while (videoIn.isOpened()):
    success, frame = videoIn.read()
    if success:
        # GET FRAME
        frame_copy = frame.copy()
        currFrame = cv2.rotate(frame_copy, cv2.cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)
        
        # APPLY CASCADE CLASSIFIER
        low = low_cascade.detectMultiScale(gray, 1.1, 3)
        
        # FIND AND TRACK POTENTIAL BOXES
        for (x,y,w,h) in low:
            # STORE MIDPOINT (AVG) OF BOTTOM SIDE OF BOXES
            xvals.append(x+int(w/2))
            yvals.append(y+h)
            wvals.append(w)
            hvals.append(h)
            
        # DRAW BOTTOMMOST BOX IN GREEN AND ITS CENTER IN RED
        if not xvals:
            xmax.append(0)
            ymax.append(0)
        else:
            max_val = max(yvals)
            max_ind = yvals.index(max_val)
            xmax.append(xvals[max_ind])
            ymax.append(yvals[max_ind])
            xp = xmax[-1] - int(wvals[max_ind]/2)
            yp = ymax[-1] - hvals[max_ind]
            cv2.rectangle(currFrame, (xp, yp), (xp+wvals[max_ind], ymax[-1]), (0, 255, 0), 2)
            cv2.circle(currFrame, (xmax[-1], ymax[-1]), 10, (0, 0, 255), -1)

        # RESET
        xvals = []
        yvals = []
        wvals = []
        hvals = []
        
        # WRITE TO OUTPUT
        videoOut.write(currFrame)
        count += jumpFrames
        videoIn.set(1, count)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break

# STORE DATA POINTS
data = np.column_stack((ymax, xmax)) # flipped order to maintain consistency
np.savetxt(videoFile[:len(videoFile)-4] + "Haar.txt", data, delimiter=',', fmt='%d')

# CLEAN UP
videoIn.release()
videoOut.release()
cv2.destroyAllWindows()

finishTime = time.time()
diffTime = finishTime - initTime
print("Total runtime: %.2f min" % (diffTime/60))
