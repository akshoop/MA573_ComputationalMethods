#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW6, question 1)

"""

# importing the necessary packages
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

##############################################################################

# increment step method for Brownian Motion
def SimBMStep(T, N):
    # initialize W brownian motion array
    W = np.zeros(int(N))
    W = np.append(W, 0)    # initial 0 value
    
    # loop
    num = 0
    while num < N:
        W[num + 1] = W[num] + sqrt(T/N)*np.random.standard_normal()
        num += 1
    
    
    return W
    
# initial vars
rho1 = 0.9
rho2 = 0.3
rho3 = 0
rho4 = -0.3
rho5 = -0.9
# setting up t x-axis variable
t = np.arange(0.0, 1001.0, 1.0)

W1 = SimBMStep(1, 1000.)
Z = np.random.standard_normal(1001)

plt.clf()

###################################       
plt.subplot(2,3,1)

# W2 part i
W2_i = rho1*W1 + sqrt(1 - rho1**2)*Z

plt.plot(t,W1, color = 'blue', label = 'W1')
plt.plot(t,W2_i, color = 'red', alpha = 0.6, label = 'W2')
plt.title("W1 vs W2 (rho = 0.9)")

# this is to label the x-axis as 0 to 1 time t
ticks = np.arange(t.min(), t.max() + 1, 1000)
labels = range(ticks.size)
plt.xticks(ticks, labels)
plt.xlabel("Time t")
plt.ylabel("Brownian Motion val")
plt.legend()

###################################       
plt.subplot(2,3,2)

# W2 part ii
W2_ii = rho2*W1 + sqrt(1 - rho2**2)*Z

plt.plot(t,W1, color = 'blue', label = 'W1')
plt.plot(t,W2_ii, color = 'red', alpha = 0.6, label = 'W2')
plt.title("W1 vs W2 (rho = 0.3)")

# this is to label the x-axis as 0 to 1 time t
ticks = np.arange(t.min(), t.max() + 1, 1000)
labels = range(ticks.size)
plt.xticks(ticks, labels)
plt.xlabel("Time t")
plt.ylabel("Brownian Motion val")
plt.legend()

###################################       
plt.subplot(2,3,3)

# W2 part iii
W2_iii = rho3*W1 + sqrt(1 - rho3**2)*Z

plt.plot(t,W1, color = 'blue', label = 'W1')
plt.plot(t,W2_iii, color = 'red', alpha = 0.6, label = 'W2')
plt.title("W1 vs W2 (rho = 0)")

# this is to label the x-axis as 0 to 1 time t
ticks = np.arange(t.min(), t.max() + 1, 1000)
labels = range(ticks.size)
plt.xticks(ticks, labels)
plt.xlabel("Time t")
plt.ylabel("Brownian Motion val")
plt.legend()

###################################       
plt.subplot(2,3,4)

# W2 part iv
W2_iv = rho4*W1 + sqrt(1 - rho4**2)*Z

plt.plot(t,W1, color = 'blue', label = 'W1')
plt.plot(t,W2_iv, color = 'red', alpha = 0.6, label = 'W2')
plt.title("W1 vs W2 (rho = -0.3)")

# this is to label the x-axis as 0 to 1 time t
ticks = np.arange(t.min(), t.max() + 1, 1000)
labels = range(ticks.size)
plt.xticks(ticks, labels)
plt.xlabel("Time t")
plt.ylabel("Brownian Motion val")
plt.legend()

###################################       
plt.subplot(2,3,5)

# W2 part v
W2_v = rho5*W1 + sqrt(1 - rho5**2)*Z

plt.plot(t,W1, color = 'blue', label = 'W1')
plt.plot(t,W2_v, color = 'red', alpha = 0.6, label = 'W2')
plt.title("W1 vs W2 (rho = -0.9)")

# this is to label the x-axis as 0 to 1 time t
ticks = np.arange(t.min(), t.max() + 1, 1000)
labels = range(ticks.size)
plt.xticks(ticks, labels)
plt.xlabel("Time t")
plt.ylabel("Brownian Motion val")
plt.legend()