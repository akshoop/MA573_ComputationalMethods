#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:11:51 2017

@author: SHOOP
"""

# ming's attempt of Q3
from scipy.stats import norm
import numpy as np

##################

# initial vars
T = 1.0
K = 115.

# on interval [0, 200]
# assume Smax = 2*S0
Smax = 200.
ds = Smax/1000.      # space step
M = 1000
timestep = 50.  # change this according to question part
dt = T/timestep  # time step
N = int(T/dt)

def explicit(k, smin = 0.0, smax = 200.0, sigma = 0.2, r = 0.03, strike = 115):
    A = np.array([[0.0]*1000]*1000)
    incr = float(smax)/1000
    s = np.array([incr*i for i in range(1000)])
    
    v = np.maximum(s - strike,0)
    
    A[0,0] = 0
    A[0,1] = 0
    A[0,2] = 0
    A[999,997] = 1
    A[999,998] = -2
    A[999,999] = 1

    for i in range(1,999):
        A[i,i] = -sigma**2*s[i]**2/incr**2 - r
        A[i,i-1] = 0.5*sigma**2*s[i]**2/incr**2 - 0.5*r*s[i]/incr
        A[i,i+1] = 0.5*sigma**2*s[i]**2/incr**2 + 0.5*r*s[i]/incr

    for j in range(int(k)):
        v = np.dot(np.identity(1000) + 1.0/k*A, v)
        v[0] = 0
        v[-1] = (smax - strike)*np.exp((j+1)/k)
    return v

x = explicit(50.)
print(x)
