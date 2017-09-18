#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW5, question 1)

"""

# importing the necessary packages
import numpy as np
from numpy import exp, log, sqrt, pi

##############################################################################

# initial var
N = 100000

# defining our g(X) function, aka Y = indicator of (X >= a)
def funcY(x, a):
    if x >= a:
        return 1
    else:
        return 0

# N sample points of X and Y rv
X_vals = np.random.standard_normal(N)

# let a = 3
a_1 = 3

Y_vals1 = []
for i,X_value in enumerate(X_vals):
    Y_vals1.append(funcY(X_value, a_1))

# b_optimal value which was calculated from Q1 part a
b_optimal_1 = 1/sqrt(2*pi)*exp(-a_1**2 /2.)

# new sampling, using the control variate
# note, E{X] = G is always 0, because X is standard normal rv.
ControlVariateEstimator1 = Y_vals1 - b_optimal_1 * (X_vals - 0)

ValueEstimation_1 = np.mean(ControlVariateEstimator1)

print("The estimation value (with a = 3) is: ")
print(ValueEstimation_1)

#####################################
# let a = 8
a_2 = 8
Y_vals2 = []
for i,X_value in enumerate(X_vals):
    Y_vals2.append(funcY(X_value, a_2))
    
# b_optimal value which was calculated from Q1 part a
b_optimal_2 = 1/sqrt(2*pi)*exp(-a_2**2 /2.)

# new sampling, using the control variate
# note, E{X] = G is always 0, because X is standard normal rv.
ControlVariateEstimator2 = Y_vals2 - b_optimal_2 * (X_vals - 0)

ValueEstimation_2 = np.mean(ControlVariateEstimator2)

print("The estimation value (with a = 8) is: ")
print(ValueEstimation_2)