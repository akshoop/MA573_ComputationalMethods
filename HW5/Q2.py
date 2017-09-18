#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW5, question 2)

"""

# importing the necessary packages
import numpy as np
from numpy import exp, log, sqrt, pi
from scipy.stats import norm

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

# mu_optimal was calculated in Question 2, part a
dens_1 = 1/sqrt(2*pi) * exp( -2*(a_1**2))
mu_optimal_1 = a_1 - (2*a_1*(1 - norm.cdf(2*a_1)) - dens_1)/((4*a_1**2 + 2)*(1 - norm.cdf(2*a_1)) - 2*a_1*dens_1)

# shift the X values, because now we're in new measure P_~
X_vals_shifted1 = np.random.normal(mu_optimal_1, 1, N)

# calculate f(X)/g(X) ratio
ratio = exp(-mu_optimal_1*X_vals_shifted1 + (mu_optimal_1**2)/2.)

# calculate Importance Sampling estimator
IS_estimator1 = Y_vals1*ratio

ValueEstimation_1 = np.mean(IS_estimator1)

print("The estimation value (with a = 3) is: ")
print(ValueEstimation_1)


#####################################
# let a = 3
a_2 = 8
Y_vals2 = []
for i,X_value in enumerate(X_vals):
    Y_vals2.append(funcY(X_value, a_2))

# mu_optimal was calculated in Question 2, part a
dens_2 = 1/sqrt(2*pi) * exp( -2*(a_2**2))
mu_optimal_2 = a_2 - (2*a_2*(1 - norm.cdf(2*a_2)) - dens_2)/((4*a_2**2 + 2)*(1 - norm.cdf(2*a_2)) - 2*a_2*dens_2)

# shift the X values, because now we're in new measure P_~
X_vals_shifted2 = np.random.normal(mu_optimal_2, 1, N)

# calculate f(X)/g(X) ratio
ratio = exp(-mu_optimal_2*X_vals_shifted2 + (mu_optimal_2**2)/2.)

# calculate Importance Sampling estimator
IS_estimator2 = Y_vals2*ratio

ValueEstimation_2 = np.mean(IS_estimator2)

print("The estimation value (with a = 8) is: ")
print(ValueEstimation_2)