#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW2, question 5)

"""

# importing the necessary packages
import numpy as np
from numpy import pi, sin, cos, sqrt, exp, log
import matplotlib.pyplot as plt
import timeit

##############################################################################
# Box Muller method

# starting timer
start = timeit.default_timer()

plt.clf()

# exact normal density function
def f(x): 
    return 1/(sqrt(2 * pi)) * exp(-(x)**2 / 2 )

# initialize array of X rv's
X1 = np.zeros(0)
X2 = np.zeros(0)

# defining Box Muller function producting std normal variables
def BoxMull(X1, X2):
    n = 0
    while n < 50000:
        
        U1 = np.random.rand()
        U2 = np.random.rand()
      
        X1 = np.append(X1, cos(2*pi*U1) * sqrt(-2*log(U2)))
        X2 = np.append(X2, sin(2*pi*U1) * sqrt(-2*log(U2)))
      
        n += 1
      
    return X1, X2

# get function values
X1, X2 = BoxMull(X1, X2)

# setting up variables to plot original density
Z = np.arange(-3, 3, 0.1)
W = f(Z)*5000

# standard normal distribution plot
plt.plot(Z, W, color = 'red')

# histogram sample plot
plt.hist([X1, X2], range = (-3, 3), bins = 60, alpha = 0.5)
plt.title ("Box-Muller method sampling")
plt.xlabel ("x Value")
plt.ylabel ("Density * 5000")

plt.show()

# stopping timer
stop = timeit.default_timer()

print "Box Muller method runtime = " , stop - start, " seconds"