#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW3, question 4)

"""

# importing the necessary packages
import numpy as np

##############################################################################

# initial variables
mu = np.array([[-1000],
                      [-700],
                      [300],
                      [-200]])

covariance = np.array([[144, 72, 120, 300],
                      [72, 100, 180, 230],
                      [120, 180, 389, 880],
                      [300, 230, 880, 4469]])

weights = np.array([0.4, 0.2, 0.3, 0.1])

# setup L losses simulation
L = np.zeros(0)

# get the matrix A thanks to built-in cholesky function
A_cholesky = np.linalg.cholesky(covariance)
    
# simulation loop for 10,000 samples
num = 0
while num < 10000:
    
    # this X matrix will hold our Assets
    X = np.zeros_like(mu)

    # creating standard normal random samples matrix
    # each element is a different standard normal random sample
    Z = np.zeros_like(A_cholesky)
    for x in np.nditer(Z, op_flags = ['readwrite']):
        x[...] = np.random.standard_normal()

    
    # loop through to get assets vector
    for i in xrange(X.size):
        X[i] = mu[i] + sum(A_cholesky[i]*Z[i])
    
    # loss based on calculation
    L = np.append(L, weights.dot(X))
    
    num += 1

# sort L array, from largest to smallest
L = np.sort(L)
L[:] = L[::-1]

# we want 99% Value At Risk. So pick the 10000*0.01 = 100th loss element

print("Numerical estimation of 99% Value At Risk is: ")
print(L[100])