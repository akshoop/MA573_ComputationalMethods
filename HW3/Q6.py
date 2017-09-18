#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW3, question 6)

"""

# importing the necessary packages
import numpy as np
from numpy import sqrt, exp
import matplotlib.pyplot as plt
import timeit

##############################################################################

# starting timer
start = timeit.default_timer()

# initial variables
S_0 = 100.
B_0 = 1
mu = 0.17
sigma = 0.25
r = 0.02

# We want to price an up-and-out barrier put option
T = 2.
K = 105.
B = 180.

# increment step method defined function from Q5_scaling.py file
def SimBMStep(T, N):
    # initialize W brownian motion array
    W = np.zeros(N)
    W = np.append(W, 0)    # initial 0 value
    
    # loop
    num = 1
    while num <= N:
        W[num] = W[num - 1] + sqrt(T/N)*np.random.standard_normal()
        num += 1
    
    
    return W
    
# stock generation defined function from Q1.py file
# note that we're multiplying by W value, instead of sqrt(T)*random normal
def generateStock(S_0, r, sigma, T, W):
    return S_0 * exp((r - (sigma**2)/2)*T + sigma * W)
    
# defining payoff function, for this case it's regular European put
def payoff_func(S, K):
    return max(K - S, 0)
###################################   

plt.clf()

# note, stepsize is 1/5000, but since T = 2, step size N = 10,000

# setting up t x-axis variable
t = np.arange(0.0, 10001.0, 1.0)

# keeping track of payoffs of up-and-out barrier put
payoffs = np.zeros(0)

# tally for number of times barrier was hit
countBarrier = 0

# loop for 10,000 different paths, for stepsize = 1/5000
n = 0
while n < 10000:
    # simulate brownian motion
    W = SimBMStep(1., 10000)
    
    # generate future stock, using the simulated W value
    S_T = generateStock(S_0, r, sigma, T, W)
    plt.plot(t, S_T, alpha = 0.5)
    
    # check if barrier condition was hit
    if np.any(S_T > B):
        # if past barrier at ANY point in time previously, payoff = 0
        payoffs = np.append(payoffs, 0)
        # keep track of how many times we hit barrier condition
        countBarrier += 1
    else:
        payoffs = np.append(payoffs, exp(-r*T)*payoff_func(S_T[S_T.size - 1], K))
    
    
    n += 1

# displaying the simulation brownian motion W
    
plt.title("Brownian Motion simulated (stepsize = 1/5000)")

ticks = np.arange(t.min(), t.max() + 1, 5000)
labels = range(ticks.size)
plt.xticks(ticks, labels)
plt.xlabel("Time t")

plt.ylabel("S_t")

plt.show()

# calculating price
MonteCarloPrice = sum(payoffs) / 10000
print("Monte Carlo integration estimation of price is: ")
print(MonteCarloPrice)
print("And number of times barrier was hit were: ")
print(countBarrier)

# stopping timer
stop = timeit.default_timer()

print "Barrier 120 method runtime = " , stop - start, " seconds"

