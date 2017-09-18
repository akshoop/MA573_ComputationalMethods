#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW4, question 4)

"""

# importing the necessary packages
import numpy as np
from numpy import exp, log, sqrt
import matplotlib.pyplot as plt
import math

##############################################################################
#part a)

plt.clf()

# variables given
S_0 = 100.

mu = 0.1
sigma = 0.2

T = 10
N = 10000.   #10 years, and 1000 steps per year
dt = T/N

lambda_var = 2.

#simulation runs
runs = 1000.

# our large Y array which will have 1000 samples
Y_stratified = np.zeros(0)

# inverse CDF of exponential
def InverseExp(x):
    y = - 0.5 * (log(1 - x))
    return y    

# Need to get stratified intervals
a_stratPoints = np.zeros(0)
for i in range(10):
    val = (i+1)/float(10)
    a_stratPoints = np.append(a_stratPoints, InverseExp(val))   

U1 = np.random.random_sample(100)
Y_stratified = np.append(Y_stratified, 0 + (a_stratPoints[0] - 0)*U1 )   

# loop for Yi values
for j in range(9):
    U2 = np.random.random_sample(100)
    
    Y_stratified = np.append(Y_stratified, a_stratPoints[j] + (a_stratPoints[j+1] - a_stratPoints[j])*U2)


# plot and info for regular density of exponential, and non-stratified sampling
x = np.random.random_sample(1000) 
y = InverseExp(x)
z = np.arange(0, 3, 0.1)
w = lambda_var * np.exp(-z * lambda_var)*100

plt.plot(z, w, color = 'red', label = 'Exp density')
plt.hist(y, range = (0,3), bins = 30, color = 'blue', label = 'Non-stratified sampling') 
plt.hist(Y_stratified, range = (0,3), bins = 30, color = 'yellow', label = 'Stratified sampling')

plt.title ("Exponential Histogram")
plt.xlabel ("x value")
plt.ylabel ("Density x 1000")

plt.legend()
plt.show()

#############################################################################
#part b)

# array to keep ST terminal stock values, for expectation calculation
ST_array = np.zeros(0)

num = 0
while num < runs:
    S_t = np.zeros(0)
    BM_sum = 0
    
    # calculating St array
    k = 0
    tau = Y_stratified[num]
    
    # if no default occured, then set time to final terminal time (aka. T = 10)
    if tau > 1.0:
        tau = 1.0
        
    # St simulation loop, until time tau.
    while k < tau:
        # have to make new Brownian Motion for each increment step
        Z = np.random.standard_normal()
        BM_increment = sqrt(dt) * Z
        BM_sum += BM_increment
        
        S_t = np.append(S_t, S_0 * exp((mu - 0.5*(sigma**2)*(k+1)*(dt) + sigma * BM_sum)))
        k += 0.001    #1000 steps each year

    ST_array = np.append(ST_array, S_t[S_t.size - 1])
    num += 1

PriceAtDefault = sum(ST_array) / ST_array.size

print("The expected price at default is: ")
print(PriceAtDefault)
