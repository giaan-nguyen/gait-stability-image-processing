#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXTRACT LINE
    1) From the terminal, input "python ExtractLine.py <videoFile>"
    2) An outputted JPEG shows the line extracted.
    3) An outputted text file stores the indices of the edges of the line.
"""
import numpy as np
import cv2, os, sys

# LOAD IN VIDEO
videoFile = sys.argv[1]
videoIn = cv2.VideoCapture(videoFile)

# GET DIMENSIONS
W = int(videoIn.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(videoIn.get(cv2.CAP_PROP_FRAME_HEIGHT))

# READ THE FIRST 10 FRAMES
success,image = videoIn.read()
count = 0
while success:
    if count < 10:
        cv2.imwrite("frame%d.jpg" % count, image)
        success,image = videoIn.read()
        print('Read a new frame: ', success)
        count += 1
    else:
        break
videoIn.release()

# AVERAGE THE 10 FRAMES TO REDUCE NOISE
init = np.zeros((H,W))
for frame in range(10):
    image = cv2.imread("frame%d.jpg" % frame)
    init += image[:,:,1] # comp 1 is green
init /= 10

# THRESHOLD GREEN VALUES
thresh = 120 # vals between 110 and 130 seem to work
init[init < thresh] = 0

# APPLY POSITIONAL MASK TO ZERO OUT SECTIONS
# Zero out left half (or top in portrait view)
offset = 0
init[:,:int(W/2)-offset] = 0 # CHANGE INDICES IF END OF LINE NOT AT THE CENTER
# Zero out top third and bottom third (or left and right in portrait view)
init[:int(H/3),:] = 0
init[-int(H/3):,:] = 0

# APPLY OPENING USING LONG RECTANGULAR ELEMENT
size = (5,50)
kernel = np.ones(size)
opening = cv2.morphologyEx(init, cv2.MORPH_OPEN, kernel)

# APPLY CANNY EDGE DETECTION
low_threshold = 80
high_threshold = 200
edges = cv2.Canny(np.uint8(opening), low_threshold, high_threshold)

# STORE "OUTLINE" PIXELS
nonzero = (edges != 0)
line_x, line_y = np.where(nonzero)
#data = np.column_stack((line_x, line_y))
#np.savetxt(videoFile[:len(videoFile)-4] + "Line.txt", data, delimiter=',', fmt='%d')

# STORE COL_VAL, LOWER_ROW, UPPER_ROW, RATIO
cols = []
rows_l = []
rows_u = []
ratios = []
for i in range(min(line_y), max(line_y) + 1):
    cols.append(i)
    ind = np.where(line_y == i)
    a = min(line_x[ind])
    b = max(line_x[ind])
    rows_l.append(a)
    rows_u.append(b)
    ratios.append( abs(b-a) )

a = np.mean(ratios[:5])
b = max(ratios)
ratios = np.linspace(a, b, len(ratios))
data = np.column_stack((cols, rows_l, rows_u, ratios))
np.savetxt(videoFile[:len(videoFile)-4] + "Line.txt", data, delimiter=',', fmt='%.2f')

image = cv2.imread("frame0.jpg")
for i in range(-3,4):
    image[line_x+i,line_y+i,:] = (255,0,0)
rotImg = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite(videoFile[:len(videoFile)-4] + "Line.jpg", rotImg)

# CLEAN UP
for i in range(10):
    os.remove("frame%d.jpg" % i)

