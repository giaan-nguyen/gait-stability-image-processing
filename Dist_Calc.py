#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DISTANCE CALCULATIONS
    1) From the terminal, input "python Dist_Calc.py <edgeTxt> <frameDiffTxt>"
    2) An outputted text file stores the footstep-to-line distances in centimeters.
    3) The terminal prints the mean and standard deviation of the distances.
"""
import numpy as np
import sys

# LOAD IN FILES
edges = np.loadtxt(sys.argv[1], delimiter=",")
data = np.loadtxt(sys.argv[2], delimiter=",")

# GET EXTREMES OF LINE
a = int(edges[0,0])
b = int(edges[-1,0])

# GET PIXEL DISTANCE BETWEEN RECORDED DATA AND LINE
dist = np.zeros((len(data), 1))
for i in range(len(data[:,0])):
    if data[i,0] in range(a, b+1): # if footstep runs parallel to line
        ind = int(data[i,0] - a)
        c = int(edges[ind,1])
        d = max( int(edges[ind,2]), int(edges[ind,1] + edges[ind,3]) )
        if data[i,1] in range(c, d+1): # if footstep falls on the line
            dist[i] = 0
        else:
            r = edges[ind,3]
            R = edges[-1,3] # baseline
            m = (r+R)/2
            dist[i] = min( abs(data[i,1] - c), abs(data[i,1] - d) ) / m
    else:
        dist[i] = 0 # not on or parallel to line yet

# TAPE IS ABOUT 5 CM IN WIDTH
centi = dist*5

file = sys.argv[2]
np.savetxt("Dist_Calc_" + file[:len(file)-4] + ".txt", centi, fmt='%.2f')
print("Mean is %.2f cm" % np.mean(centi))
print("Standard deviation is %.2f cm" % np.std(centi))
