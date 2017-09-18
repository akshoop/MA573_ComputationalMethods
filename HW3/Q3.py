#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW3, question 3)

"""

# importing the necessary packages
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

##############################################################################

# initial variables
covariance = np.zeros([2,2])
sigma1 = 3.2
sigma2 = 0.7
corr = 0.6
mu1 = 2
mu2 = 3

# setting up array for actual covariance calculation
array1 = np.array([[sqrt(sigma1), 0],
                   [0, sqrt(sigma2)]])
array2 = np.array([[1, corr],
                   [corr, 1]])

covariance = (array1.dot(array2)).dot(array1)
print("The covariance matrix is: ")
print(covariance)

##############################################################################

# hand-written result of cholesky factorization of A
# A = [[1.78885, 0]
#      [0.501996, 0.669328]]

# cholesky factorization, from built-in default function
A_actual = np.linalg.cholesky(covariance)
print("The matrix A, based on built-in cholesky function np.linalg.cholesky(): ")
print(A_actual)

##############################################################################

# cholesky factorization by python algorithm coding

# Note, need to do A_actual.tolist() to make it python list type

def cholesky(A):
    # establish output S matrix
    n = len(A)
    
    # setup output array
    S = [[0.0] * n for i in range(n)]

    # setting up calculation of individual elements for loop
    for i in range(n):
        for j in range(i + 1):
            # sigma_ij formula from class
            temp = sum(S[i][k] * S[j][k] for k in range(j))
            
            # for the diagnoal case
            if i == j:
                S[i][j]= sqrt(A[i][i] - temp)
            # for regular case
            else:
                S[i][j] = (A[i][j] - temp) / S[j][j]

    return S
    
A_cholesky = np.array(cholesky(covariance))

print("The matrix A, based on python code algorithm function: ")
print(A_cholesky)

##############################################################################

# scatterplot work

plt.clf()

num = 0

# producing 1000 samples
while num < 1000:
    
    X1 = mu1 + A_cholesky[0,0]*np.random.standard_normal()
    X2 = mu2 + A_cholesky[1,0]*np.random.standard_normal() + A_cholesky[1,1]*np.random.standard_normal()
    
    plt.scatter(X1, X2)
    num += 1

plt.title("Scatterplot of (X1,X2)^T random vector samples")
plt.xlabel("X1")
plt.ylabel("X2")

plt.show()


