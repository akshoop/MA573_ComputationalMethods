#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW6, question 5)

"""

# importing the necessary packages
import numpy as np
from numpy import sqrt, exp, log
import matplotlib.pyplot as plt

##############################################################################

# increment step method for Brownian Motion
def SimBMStep(T, N):
    # initialize W brownian motion array
    W = np.zeros(int(N))
    W[0] = 0    # initial 0 value

    # loop
    num = 0
    while num < N - 1:
        W[num + 1] = W[num] + sqrt(T/N)*np.random.standard_normal()
        num += 1

    return W

# initial vars
y = 0.08
lam = 3.
kappa = 0.1
xi = 0.25

T = 1
N = 100.

t = np.arange(0, 1.0, 0.01)

i = 0
while i < 10:
    W1 = SimBMStep(T, N)
    
    Y = np.zeros_like(W1)
    X = np.zeros_like(W1)
    Y[0] = y
    X[0] = sqrt(y)
    temp = 0
    while temp < X.size-1:
        X[temp + 1] = (xi*(W1[temp+1] - W1[temp]) 
                        + sqrt(xi**2 * (W1[temp+1] - W1[temp])**2 
                               + 4*(1+lam*(T/N))*(X[temp]**2 + (T/N)*(lam*kappa - (xi**2/2)))))/(2*(1+lam*(T/N)))
        Y[temp + 1] = X[temp+1]**2
        temp += 1
    plt.plot(t, Y)
    i += 1

plt.title("Simulated CIR Yt paths")
plt.xlabel("Time t")
plt.ylabel("Process value")
plt.show()
