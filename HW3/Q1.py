#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW3, question 1)

"""

# importing the necessary packages
import numpy as np
from numpy import sqrt, exp, log
from scipy.stats import norm

##############################################################################

# variables given
S_0 = 100.
B0 = 1
mu = 0.03
sigma = 0.25
r = 0.01

# European Call, using Black Scholes formula
K = 120.
T = 1

d1 = (log(S_0/K) + (r + (sigma**2)/2)*T) / (sigma * sqrt(T))
d2 = d1 - sigma*sqrt(T)

Price1 = S_0*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

print("European Call option price (using Black Scholes formula) is:")
print(round(Price1,2))

##############################################################################
# question 1)

# defining the stock generation function
def generateStock(S_0, r, sigma, T):
    return S_0 * exp((r - (sigma**2)/2)*T + sigma * sqrt(T) * np.random.standard_normal())
    
# defining payoff function, for this case it's regular European call
def payoff1(S, K):
    return max(S - K, 0)
 
# defining monte carlo integration formula, to make graphing easier
def MonteCarlo1(runs):
    # initialize array that will have payoffs of option
    payoffs = np.zeros(0)

    # looping through
    for i in xrange(runs):
        # generate future stock
        S_T = generateStock(S_0, r, sigma, T)
    
        # append to the payoffs list whatever the payoff is
        payoffs = np.append(payoffs,
                            exp(-r*T)*payoff1(S_T, K))
    
    return sum(payoffs)/runs, payoffs

# setting number of simulation runs
runs = 10000

 # initialize array that will have payoffs of option
payoffs = np.zeros(0)

# monte carlo calculation of option price
Price2, payoffs = MonteCarlo1(runs)

var_sample = 0
payoffs_int = payoffs.astype(int)
# setting up iterating array
it = np.nditer(payoffs, flags = ['f_index'])

while not it.finished:
    var_sample = var_sample + (payoffs[it.index] - Price2)**2
    it.iternext()
var_sample = var_sample / (runs - 1)


# from analytical calculation, we know sigma = Var(X1) = 657.973

# 95% confidence, z-score = 1.96
size_estimate = (1.96**2 * 657.973) / 0.05**2

print("European Call option price (using Monte Carlo integration) is: ")
print(round(Price2,2))
print("Approximation of how many samples needed for 95% confidence (up to dime):")
print(int(size_estimate))

# running Monte Carlo calculation 100 times with the above size estimate
# counters for counting number of times the Monte Carlo estimated price
# correct up to the dime or not
correct = 0
incorrect = 0
for j in xrange(100):
    price_to_compare, dummypayoff = MonteCarlo1(int(size_estimate))
    
    if price_to_compare > Price1 - 0.05 and price_to_compare < Price1 + 0.05:
        correct += 1
    else:
        incorrect += 1

print("Number of times Monte Carlo estimation was within a dime:")
print(correct)
print("Number of times Monte Carlo estimation was NOT within a dime:")
print(incorrect)

