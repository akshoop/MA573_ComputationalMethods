#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW3, question 5)

"""

# importing the necessary packages
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

##############################################################################

# increment step method
def SimBMStep(T, N):
    # initialize W brownian motion array
    W = np.zeros(N)
    W = np.append(W, 0)    # initial 0 value
    
    # loop
    num = 1
    while num <= N:
        W[num] = W[num - 1] + sqrt(T/N)*np.random.standard_normal()
        num += 1
    
    
    return W

###################################   

plt.clf()
    
plt.subplot(2,2,1)

# setting up t x-axis variable
t = np.arange(0.0, 11.0, 1.0)

# loop for 10 different paths
n = 0
while n < 10:
    W = SimBMStep(1., 10)
    plt.plot(t, W, alpha = 0.5)
    n += 1
    
plt.title("Brownian Motion simulated (N = 10)")
plt.xlabel("Time t")
plt.ylabel("Brownian Motion val")

###################################   
plt.subplot(2,2,2)    

# setting up t x-axis variable
t = np.arange(0.0, 101.0, 1.0)

# loop for 10 different paths
n = 0
while n < 10:
    W = SimBMStep(1., 100)
    plt.plot(t, W, alpha = 0.5)
    n += 1
    
plt.title("Brownian Motion simulated (N = 100)")
plt.xlabel("Time t")
plt.ylabel("Brownian Motion val")

###################################       
plt.subplot(2,2,3)

# setting up t x-axis variable
t = np.arange(0.0, 1001.0, 1.0)

# loop for 10 different paths
n = 0
while n < 10:
    W = SimBMStep(1., 1000)
    plt.plot(t, W, alpha = 0.5)
    n += 1
    
plt.title("Brownian Motion simulated (N = 1000)")
plt.xlabel("Time t")
plt.ylabel("Brownian Motion val")

###################################   
plt.subplot(2,2,4)

# setting up t x-axis variable
t = np.arange(0.0, 10001.0, 1.0)

# loop for 10 different paths
n = 0
while n < 10:
    W = SimBMStep(1., 10000)
    plt.plot(t, W, alpha = 0.5)
    n += 1
    
plt.title("Brownian Motion simulated (N = 10000)")
plt.xlabel("Time t")
plt.ylabel("Brownian Motion val")

plt.show()

"""
##############################################################################

# cholesky method 

def SimBMChol(T, N):
    # initialize W brownian motion array
    W = np.zeros(N+1)
    W = np.append(W, 0)    # initial 0 value

    # looping and iterating through
    for i in xrange(N):
        Z = np.zeros(i+1)
        for x in np.nditer(Z, op_flags = ['readwrite']):
            x[...] = np.random.standard_normal()
            
        val = sum(sqrt(N/T)*Z)
        W = np.append(W, val)
        
    return W

###################################    
plt.clf()
# setting up t x-axis variable
t = np.arange(0.0, 11.0, 1.0)

plt.subplot(2, 2, 1)

num = 0
while num < 10:
    W = SimBM(1, 10)
    plt.plot(t, W, alpha = 0.5)
    num += 1

plt.title("Brownian Motion simulated (N = 10)")
plt.xlabel("Time t")
plt.ylabel("Brownian Motion val")

################################### 
plt.subplot(2, 2, 2)

# setting up t x-axis variable
t = np.arange(0.0, 101.0, 1.0)

num = 0
while num < 10:
    W = SimBM(1, 100)
    plt.plot(t, W, alpha = 0.5)
    num += 1

plt.title("Brownian Motion ten simulated paths (N = 100)")
plt.xlabel("Time t")
plt.ylabel("Brownian Motion val")

################################### 
plt.subplot(2, 2, 3)

# setting up t x-axis variable
t = np.arange(0.0, 1001.0, 1.0)

num = 0
while num < 10:
    W = SimBM(1, 1000)
    plt.plot(t, W, alpha = 0.5)
    num += 1

plt.title("Brownian Motion ten simulated paths (N = 1000)")
plt.xlabel("Time t")
plt.ylabel("Brownian Motion val")

################################### 

plt.subplot(2, 2, 4)

# setting up t x-axis variable
t = np.arange(0.0, 10001.0, 1.0)

num = 0
while num < 10:
    W = SimBM(10000, 1)
    plt.plot(t, W, alpha = 0.5)
    num += 1

plt.title("Brownian Motion ten simulated paths (N = 10000)")
plt.xlabel("Time t")
plt.ylabel("Brownian Motion val")
"""
