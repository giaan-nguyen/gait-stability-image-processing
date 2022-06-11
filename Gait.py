#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAIT ANALYSIS
    1) From the terminal, input "python Gait.py <DistL> <DistR> <DistC>"
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import sys

# LOAD IN FILES
centL = np.loadtxt(sys.argv[1], delimiter=",")
centR = np.loadtxt(sys.argv[2], delimiter=",")
centC = np.loadtxt(sys.argv[3], delimiter=",")

cent0 = np.copy(centL)
cent1 = np.copy(centR)
cent2 = np.copy(centC)

# GAIT CADENCE
peaks, _ = find_peaks(centC)

diff_pks = np.zeros(len(peaks)-1) # step times
for i in range(len(peaks)):
    if i > 0:
        diff_pks[i-1] = peaks[i] - peaks[i-1]

mean_diff = np.mean(diff_pks) # expected  step time
print("The expected step time (based on foot-sweeps) is:")
print(mean_diff/10)

count = 0
alpha = 0.2
for p in diff_pks:
    if p > (1+alpha)*mean_diff or p < (1-alpha)*mean_diff: # inconsistent step time
        count += 1

print("The step time consistency ratio (STCR) is:") # consistent-to-total step times
print(1-count/len(diff_pks))

plt.plot(np.array(range(len(centC)))/10, centC/2.54, color='g')
plt.plot(peaks/10,centC[peaks]/2.54,"x")
plt.title('OpenPose (Center)')
plt.xlabel('Time [s]')
plt.ylabel('Distance Gap [in]')
plt.show()

# LINEAR STEADINESS
thresh = 8 # account for foot sweep
cent0[cent0 < thresh] = 0
cent1[cent1 < thresh] = 0
cent2[cent2 < thresh] = 0

cent_mean = np.mean(cent2)
print("The left, right, and center means are:")
print(np.mean(cent0),np.mean(cent1),cent_mean)
print("The steady-to-total instance ratio (STIR) is:") 
print(len(cent2[cent2 <= cent_mean])/len(cent2))

plt.plot(np.array(range(len(cent0)))/10, cent0/2.54, color='b')
plt.plot(np.array(range(len(cent1)))/10, cent1/2.54, color='r')
plt.plot(np.array(range(len(cent2)))/10, cent2/2.54, color='g')
plt.plot(np.array(range(len(cent2)))/10, cent_mean/2.54*np.ones(len(cent2)), color='y')
plt.legend(['Camera Left', 'Camera Right', 'Center'])
plt.title('OpenPose')
plt.xlabel('Time [s]')
plt.ylabel('Distance Gap [in]')
plt.show()

