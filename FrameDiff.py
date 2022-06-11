#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMPLEMENTATION OF FRAME DIFFERENCING METHOD
    1) From the terminal, input "python FrameDiff.py <videoFile>"
    2) An outputted video file tracks the bottommost changes in the video.
    3) An outputted text file stores the centers of the bottommost changes.
"""
import numpy as np
import cv2, os, sys, time

initTime = time.time()

def getBackground(videofile):
    cap = cv2.VideoCapture(videofile)
    # SELECT 50 RANDOM FRAMES
    frame_ind = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)
    frames_list = []
    for i in frame_ind:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _,frame = cap.read()
        frames_list.append(frame)
    # FIND MEDIAN
    output = np.median(frames_list, axis=0).astype(np.uint8)
    return output

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
consec = 3

# OUTPUT VIDEO
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
videoOut = cv2.VideoWriter("diff_" + videoFile, fourcc, fps/consec, (int(H),int(W)))

# BACKGROUND MODEL
background = getBackground(videoFile)
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# BEGIN READING
while (videoIn.isOpened()):
    success, frame = videoIn.read()
    if success:
        # GET FRAME
        count += 1
        currFrame = frame.copy()
        gray = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)
        if count % consec == 0 or count == 1:
            frame_diff_list = []
    
        # FRAME DIFFERENCE OF CURRENT AND BACKGROUND
        diff = cv2.absdiff(gray, background)
        _,thres = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
        
        # APPLY DILATION
        dil = cv2.dilate(thres, None, iterations=2)
        frame_diff_list.append(dil)
        
        # SUM CONSECUTIVE FRAMES
        if len(frame_diff_list) == consec:
            sum_frames = sum(frame_diff_list)
            contours, _ = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # FIND AND TRACK POTENTIAL BOXES
            for (i,_) in enumerate(contours):
                cv2.drawContours(frame, contours, i, (0, 0, 255), 3)
            for contour in contours:
                if cv2.contourArea(contour) < 500:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                # STORE MIDPOINT (AVG) OF BOTTOM SIDE OF BOXES
                xvals.append(x+w)
                yvals.append(y+int(h/2))
                wvals.append(w)
                hvals.append(h)
            
            # DRAW REDMOST BOX IN GREEN AND ITS CENTER IN RED
            max_val = max(xvals)
            max_ind = xvals.index(max_val)
            xmax.append(xvals[max_ind])
            ymax.append(yvals[max_ind])
            xp = xmax[-1] - wvals[max_ind]
            yp = ymax[-1] - int(hvals[max_ind]/2)
            cv2.rectangle(currFrame, (xp, yp), (xmax[-1], yp+hvals[max_ind]), (0, 255, 0), 2)
            cv2.circle(currFrame, (xmax[-1], ymax[-1]), 10, (0, 0, 255), -1)

            # RESET
            xvals = []
            yvals = []
            wvals = []
            hvals = []
            
            rotFrame = cv2.rotate(currFrame, cv2.cv2.ROTATE_90_CLOCKWISE)
            # WRITE TO OUTPUT
            videoOut.write(rotFrame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    else:
        break

# STORE DATA POINTS
data = np.column_stack((xmax, ymax))
np.savetxt(videoFile[:len(videoFile)-4] + "Diff.txt", data, delimiter=',', fmt='%d')

# CLEAN UP
videoIn.release()
videoOut.release()
cv2.destroyAllWindows()

# FIX ORIENTATION
#os.system("ffmpeg -i out.mp4 -vf transpose=1 output.mp4")
#os.remove("out.mp4")
#os.rename("output.mp4", "diff_" + videoFile)

finishTime = time.time()
diffTime = finishTime - initTime
print("Total runtime: %.2f min" % (diffTime/60))
