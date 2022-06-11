#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMPLEMENTATION OF OPENPOSE BODY_25 METHOD
    1) From the terminal, input "python OpenPoseDNN.py <videoFile>"
    2) An outputted video file tracks changes in the skeleton.
    3) An outputted text file stores the camera left/right ankle positions and their centers.
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
camLeft = []
camRight = []

# INITIALIZE
count = 0
jumpFrames = 3
inWidth = 368
inHeight = 368
threshold = 0.1
num_points = 25
point_pairs = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6],
               [6, 7], [0, 15], [15, 17], [0, 16], [16, 18], [1, 8],
               [8, 9], [9, 10], [10, 11], [11, 22], [22, 23], [11, 24],
               [8, 12], [12, 13], [13, 14], [14, 19], [19, 20], [14, 21]]

# LOAD IN MODEL
prototxt = "models/pose/body_25/pose_deploy.prototxt"
caffemodel = "models/pose/body_25/pose_iter_584000.caffemodel"
body_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

# OUTPUT VIDEO
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
videoOut = cv2.VideoWriter("OP_" + videoFile, fourcc, fps/jumpFrames, (int(H),int(W)))

# BEGIN READING
while (videoIn.isOpened()):
    success, frame = videoIn.read()
    if success:
        # GET FRAME
        frame_copy = frame.copy()
        currFrame = cv2.rotate(frame_copy, cv2.cv2.ROTATE_90_CLOCKWISE)
        
        # PREDICT
        img_height, img_width, _ = currFrame.shape
        inpBlob = cv2.dnn.blobFromImage(currFrame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        body_model.setInput(inpBlob)
        body_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        body_model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        output = body_model.forward()
        h = output.shape[2]
        w = output.shape[3]

        # FIND POINTS
        points = []
        for idx in range(num_points):
            probMap = output[0, idx, :, :] # confidence map.
            
            # Find global maxima of the probMap.
            _, prob, _, point = cv2.minMaxLoc(probMap)
            
            # Scale the point to fit on the original image
            x = (img_width * point[0]) / w
            y = (img_height * point[1]) / h
            
            if prob > threshold:
                points.append((int(x), int(y)))
            else:
                points.append(None)

        # Draw Skeleton
        for pair in point_pairs:
            partA = pair[0]
            partB = pair[1]
            
            if points[partA] and points[partB]:
                cv2.line(currFrame, points[partA], points[partB], (0, 255, 255), 3)
                cv2.circle(currFrame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                if partA == 10: # identified tibia/fibula on camera left
                    camLeft.append(points[partB])
                elif partA == 13: # camera right
                    camRight.append(points[partB])
            else:
                if partA == 10:
                    if not camLeft:
                        camLeft.append((0,0)) # if empty, set 0
                    else:
                        camLeft.append(camLeft[-1]) # take previous value
                elif partA == 13:
                    if not camRight:
                        camRight.append((0,0)) # if empty, set 0
                    else:
                        camRight.append(camRight[-1]) # take previous
    
        # WRITE TO OUTPUT
        videoOut.write(currFrame)
        count += jumpFrames
        videoIn.set(1, count)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break

# STORE DATA POINTS
camLeft = np.array(camLeft)
camLeft[:,[0, 1]] = camLeft[:,[1, 0]]
camRight = np.array(camRight)
camRight[:,[0, 1]] = camRight[:,[1, 0]]
camCent = (camLeft + camRight)/2
np.savetxt(videoFile[:len(videoFile)-4] + "OP_L.txt", camLeft, delimiter=',', fmt='%d')
np.savetxt(videoFile[:len(videoFile)-4] + "OP_R.txt", camRight, delimiter=',', fmt='%d')
np.savetxt(videoFile[:len(videoFile)-4] + "OP_C.txt", camCent, delimiter=',', fmt='%d')

# CLEAN UP
videoIn.release()
videoOut.release()
cv2.destroyAllWindows()

finishTime = time.time()
diffTime = finishTime - initTime
print("Total runtime: %.2f min" % (diffTime/60))
