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

##############################################################################
# Box Muller method

plt.clf()

# exact normal density function
def f(x): 
    return 1/(sqrt(2 * pi)) * exp(-(x)**2 / 2 )

# defining Box Muller function producting std normal variables
def BoxMull(U1, U2):
  X1 = cos(2*pi*U1) * sqrt(-2*log(U2))
  X2 = sin(2*pi*U1) * sqrt(-2*log(U2))
  return X1, X2

# defining Marsaglia Bray function producting std normal variables
def MarsBray(U1, U2):
    # array of accepted variables
    a = np.zeros(0)
    
    # setting V variables
    V1 = 2*U1 - 1
    V2 = 2*U2 - 1
    
    # value to check
    S = V1**2 + V2**2
    
    if S < 1:
        # accept
        X1 = V1 * sqrt((-2 * log(S)) / S)
        X2 = V2 * sqrt((-2 * log(S)) / S)
        a = np.append(a, (X1, X2))
    else:
        # reject
    
U1 = np.random.rand(10000)
U2 = np.random.rand(10000)

# get function values
X1, X2 = BoxMull(U1, U2)

# standard cauchy distribution
#X = np.random.standard_cauchy(10000)

# generalized inverse
# there is no built in function for cotangent.
# so I used the identity: cot(x) = 1/tan(x)
#Y = -(1./tan(pi*X))

# setting up variables to plot original density
Z = np.arange(-3, 3, 0.1)
W = f(Z)*1000

# bin size = 60

# standard normal distribution plot
plt.plot(Z, W, color = 'red')

# histogram sample plot
plt.hist([X1, X2], range = (-3, 3), bins = 60, alpha = 0.5)
plt.title ("Box-Muller vs Marsaglia-Bray method sampling")
plt.xlabel ("x Value")
plt.ylabel ("Density * 1000")

plt.show()
