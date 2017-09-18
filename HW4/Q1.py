#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW4, question 1)

"""

# importing the necessary packages
import numpy as np
from numpy import sqrt, exp, log
from scipy.stats import norm
import matplotlib.pyplot as plt

##############################################################################

plt.clf()

# variables given
S_0 = 100.
B0 = 1
mu = 0.05
sigma = 0.35
r = 0.05

# number of simulation runs/sample size for Monte Carlo
runs = 5000

# setting up sample size n x-axis variable for plotting
n = np.arange(0.0, runs, 1.0)

# European Call, using Black Scholes formula
K = 120.
T = 2

d1 = (log(S_0/K) + (r + (sigma**2)/2)*T) / (sigma * sqrt(T))
d2 = d1 - sigma*sqrt(T)

# black scholes formula price calculation
Price1 = S_0*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

print("European Call option price (using Black Scholes formula) is:")
print(round(Price1,2))

plt.axhline(y = Price1, color = 'black', linestyle = '-', label = 'BS formula')
plt.legend()

#####################################

# defining the stock generation function
def generateStock(S_0, r, sigma, T):
    return S_0 * exp((r - (sigma**2)/2)*T + sigma * sqrt(T) * np.random.standard_normal())
    
# defining payoff function, for this case it's regular European call
def payoff1(S, K):
    return max(S - K, 0)

# defining monte carlo integration formula (direct version)
def MonteCarlo_Direct(runs):
    # initialize array that will have payoffs of option
    payoffs = np.zeros(0)

    # looping through
    for i in xrange(runs):
        # generate future stock
        S_T = generateStock(S_0, r, sigma, T)
    
        # append to the payoffs list whatever the payoff is
        payoffs = np.append(payoffs,
                            exp(-r*T)*payoff1(S_T, K))
    
    return sum(payoffs)/runs

# array holding Monte Carlo direct approximation values
# this is also initial point, n = 0
MC1 = np.zeros(0)

# counter for loop
num = 1

# loop for rest of Monte Carlo approximation values, until sample size 5000
while num <= runs:
    MC1 = np.append(MC1, MonteCarlo_Direct(num))
    num += 1
    
plt.plot(n, MC1, color = 'red', alpha = 0.7, label = 'Monte Carlo (direct)')
plt.legend()

print("European Call option price (using direct Monte Carlo) is:")
print(round(MC1[MC1.size - 1],2))

#####################################

# defining monte carlo integration formula WITH antithetic method
def MonteCarlo_Anti(runs):
    # initialize array that will have payoffs of option
    payoffs1 = np.zeros(0)
    payoffs2 = np.zeros(0)

    # looping through
    for i in xrange(runs):
        # generate future stock
        S_T = generateStock(S_0, r, sigma, T)
        S_T_neg = -(generateStock(S_0, r, sigma, T))
    
        # append to the payoffs list whatever the payoff is
        payoffs1 = np.append(payoffs1, exp(-r*T)*payoff1(S_T, K))
        payoffs2 = np.append(payoffs2, exp(-r*T)*payoff1(np.abs(S_T_neg), K))
    
    sum_payoffs = (payoffs1 + payoffs2) / 2
    return np.mean(sum_payoffs)

# array holding Monte Carlo antithetic approximation
MC2 = np.zeros(0)

# counter for loop
num = 1

# loop for rest of Monte Carlo approximation values, until sample size 5000
while num <= runs:
    MC2 = np.append(MC2, MonteCarlo_Anti(num))
    num += 1
    
plt.plot(n, MC2, color = 'blue', alpha = 0.7, label = 'Monte Carlo (antithetic)')
plt.legend()

print("European Call option price (using Monte Carlo & Antithetic method) is:")
print(round(MC2[MC2.size - 1],2))



plt.title("Monte Carlo price estimation")
plt.xlabel("# of samples")
plt.ylabel("Value")

plt.show()

