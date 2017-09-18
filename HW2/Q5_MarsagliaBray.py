#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW2, question 5)

"""

# importing the necessary packages
import numpy as np
from numpy import pi, sqrt, exp, log
import matplotlib.pyplot as plt
import timeit

##############################################################################
# Marsaglia-Bray method

# starting timer
start = timeit.default_timer()

plt.clf()

# exact normal density function
def f(x): 
    return 1/(sqrt(2 * pi)) * exp(-(x)**2 / 2 )

# initialize array of accepted rv's
X1 = np.zeros(0)
X2 = np.zeros(0)

# defining Marsaglia Bray function producting std normal variables
def MarsBray(X1, X2):
    n = 0
    while n < 50000:
         # setting dummy initial value
         S = 2
         
         # this while loop makes sure that we are using an accepted S
         while S > 1:
             V1 = 2 * np.random.rand() - 1
             V2 = 2 * np.random.rand() - 1
             S = (V1 ** 2) + (V2 ** 2)
             
         X1 = np.append(X1, V1 * sqrt(-2 * log(S) / S))
         X2 = np.append(X2, V2 * sqrt(-2 * log(S) / S))
         
         n += 1
    
    return X1, X2
         
# get function values
X1, X2 = MarsBray(X1, X2)
 
# setting up variables to plot original density
Z = np.arange(-3, 3, 0.1)
W = f(Z)*5000

# standard normal distribution plot
plt.plot(Z, W, color = 'red')

# histogram sample plot
plt.hist([X1, X2], range = (-3, 3), bins = 60, alpha = 0.5)
plt.title ("Marsaglia-Bray method sampling")
plt.xlabel ("x Value")
plt.ylabel ("Density * 5000")

plt.show()

# stopping timer
stop = timeit.default_timer()

print "Marsaglia-Bray method runtime = " , stop - start, " seconds"

