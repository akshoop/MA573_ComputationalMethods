#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW2, question 3)b)

"""

# importing the necessary packages
import numpy as np
from numpy import pi, sqrt, exp
import matplotlib.pyplot as plt

##############################################################################
# acceptance rejection method

# original density function
def f(x):
    return 2./sqrt(2.*pi) * exp(-(x**2) / 2.)
# density function of dominant exponential distribution
def g(x):
    return exp(-x)

# array that will be filled with accepted variables
a = np.zeros(0)

# constant c for method
c = 2./sqrt(2.*pi) * exp(0.5)

n = 0

plt.clf()

# want 10000 samples
while n < 10000:
    U1 = np.random.rand()
    U2 = np.random.rand()
    Y = -np.log(1 - U1)
    
    if c * g(Y) * U2 <= f(Y):
        n += 1 
        # accept
        a = np.append(a, Y)
   
         
        
plt.hist(a, range = (0, 3), bins = 30, color = 'blue')

# setting up variables to plot original density
\
W = f(Z)*1000

# original distribution plot
plt.plot(Z, W, color = 'red')
plt.title ("Sample of folded normal rv")
plt.xlabel ("x Value")
plt.ylabel ("Density * 1000")

plt.show()
