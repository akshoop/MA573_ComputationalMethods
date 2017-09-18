#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW2, question 1)b)

"""

# importing the necessary packages
import numpy as np
from numpy import pi, tan
import matplotlib.pyplot as plt

##############################################################################
# CAUCHY

# density function
def f(x):
    return (1/pi) * 1/(1 + x**2)

# standard cauchy distribution
X = np.random.standard_cauchy(10000)

# generalized inverse
# there is no built in function for cotangent.
# so I used the identity: cot(x) = 1/tan(x)
Y = -(1./tan(pi*X))

# setting up variables to plot original density
Z = np.arange(-5, 5, 0.1)
W = f(Z)*1000

# standard cauchy distribution plot
plt.plot(Z, W, color = 'red')
# histogram sample plot
plt.hist(Y, range = (-5, 5), bins = 100, color = 'blue')
plt.title ("Standard Cauchy histogram")
plt.xlabel ("x Value")
plt.ylabel ("Density * 1000")

plt.show()
