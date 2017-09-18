#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW2, question 6)

"""

# importing the necessary packages
import numpy as np
from numpy import sqrt, exp, log
from scipy.stats import norm
import matplotlib.pyplot as plt

##############################################################################
# question 6)a)

# variables given
S_0 = 100.
B0 = 1
mu = 0.05
sigma = 0.2
r = 0.03

# European Put, using Black Scholes formula
K = 110.
T = 0.5

d1 = (log(S_0/K) + (r + (sigma**2)/2)*T) / (sigma * sqrt(T))
d2 = d1 - sigma*sqrt(T)

Price1 = K*exp(-r*T)*norm.cdf(-d2) - S_0*norm.cdf(-d1)

print("European Put option price (using Black Scholes formula) is:")
print(round(Price1,2))

##############################################################################
# question 6)b)

# defining the stock generation function
def generateStock(S_0, r, sigma, T):
    return S_0 * exp((r - (sigma**2)/2)*T + sigma * sqrt(T) * np.random.standard_normal())
    
# defining payoff function, for this case it's regular European put
def payoff1(S, K):
    return max(0, K - S)
    
# setting number of simulation runs
runs = 10000
# initialize array that will have payoffs of option
payoffs = np.zeros(0)

# looping through
for i in xrange(runs):
    # generate future stock
    S_T = generateStock(S_0, r, sigma, T)
    
    # append to the payoffs list whatever the payoff is
    payoffs = np.append(payoffs,
                        payoff1(S_T, K))
    
Price2 = exp(-r*T) * sum(payoffs)/runs

print("European Put option price (using Monte Carlo integration) is: ")
print(round(Price2,2))

##############################################################################
# question 6)c)
    
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
                            payoff1(S_T, K))
    
    return exp(-r*T) * sum(payoffs)/runs

################################
plt.subplot(2, 2, 1)

n = 0

while n < 50:
    y = MonteCarlo1(n)
    
    plt.plot(n, y, color = 'red', marker = '.')
    
    n += 1

plt.plot(n, y, color = 'red', marker = '.', 
         label = "samples n = 50");

plt.title ("Approximation of European Put")
plt.xlabel ("Trials")
plt.ylabel ("Value")
plt.legend()

################################
plt.subplot(2, 2, 2)

n = 0

while n < 100:
    y = MonteCarlo1(n)
    
    plt.plot(n, y, color = 'red', marker = '.')
    
    n += 1

plt.plot(n, y, color = 'red', marker = '.', 
         label = "samples n = 100");

plt.title ("Approximation of European Put")
plt.xlabel ("Trials")
plt.ylabel ("Value")
plt.legend()

################################
plt.subplot(2, 2, 3)

n = 0

while n < 500:
    y = MonteCarlo1(n)
    
    plt.plot(n, y, color = 'red', marker = '.')
    
    n += 1

plt.plot(n, y, color = 'red', marker = '.', 
         label = "samples n = 500");

plt.title ("Approximation of European Put")
plt.xlabel ("Trials")
plt.ylabel ("Value")
plt.legend()

################################
plt.subplot(2, 2, 4)

n = 0

while n < 1000:
    y = MonteCarlo1(n)
    
    plt.plot(n, y, color = 'red', marker = '.')
    
    n += 1

plt.plot(n, y, color = 'red', marker = '.', 
         label = "samples n = 1000");

plt.title ("Approximation of European Put")
plt.xlabel ("Trials")
plt.ylabel ("Value")

plt.legend()

plt.show()

##############################################################################
# question 6)d)

# European asset-or-nothing digital call option
# same maturity T = 0.5, same strike K = 110

# defining payoff function
def payoff2(S, K):
    if S > K:
        return S
    else:
        return 0

# setting number of simulation runs
runs = 10000
# initialize array that will have payoffs of option
payoffs = np.zeros(0)

# looping through
for i in xrange(runs):
    # generate future stock
    S_T = generateStock(S_0, r, sigma, T)
    
    # append to the payoffs list whatever the payoff is
    payoffs = np.append(payoffs,
                        payoff2(S_T, K))
    
Price3 = exp(-r*T) * sum(payoffs)/runs

print("European asset-or-nothing digital call option price (using Monte Carlo integration) is: ")
print(round(Price3,2))

##############################################################################
# question 6)e)

# European cubic put option
# same maturity T = 0.5, same strike K = 110

# defining payoff function
def payoff3(S, K):
    return (max(0, K - S))**3

# setting number of simulation runs
runs = 10000
# initialize array that will have payoffs of option
payoffs = np.zeros(0)

# looping through
for i in xrange(runs):
    # generate future stock
    S_T = generateStock(S_0, r, sigma, T)
    
    # append to the payoffs list whatever the payoff is
    payoffs = np.append(payoffs,
                        payoff3(S_T, K))
    
Price4 = exp(-r*T) * sum(payoffs)/runs

print("European cubic put option price (using Monte Carlo integration) is: ")
print(round(Price4,2))

##############################################################################
# question 6)f)

# European gap call option
# same maturity T = 0.5, same strike K = 110

# L exercise level
L = 105

# defining payoff function
def payoff4(S, K):
    if S > K:
        return max(0, S - L)
    else:
        return 0

# setting number of simulation runs
runs = 10000
# initialize array that will have payoffs of option
payoffs = np.zeros(0)

# looping through
for i in xrange(runs):
    # generate future stock
    S_T = generateStock(S_0, r, sigma, T)
    
    # append to the payoffs list whatever the payoff is
    payoffs = np.append(payoffs,
                        payoff4(S_T, K))
    
Price5 = exp(-r*T) * sum(payoffs)/runs

print("European gap call option price (using Monte Carlo integration) is: ")
print(round(Price5,2))

##############################################################################
# question 6)g)

# European exponential put option
# same maturity T = 0.5, same strike K = 110

# defining payoff function
def payoff5(S, K):
    return exp(max(0, K - S))

# setting number of simulation runs
runs = 10000
# initialize array that will have payoffs of option
payoffs = np.zeros(0)

# looping through
for i in xrange(runs):
    # generate future stock
    S_T = generateStock(S_0, r, sigma, T)
    
    # append to the payoffs list whatever the payoff is
    payoffs = np.append(payoffs,
                        payoff5(S_T, K))
    
Price6 = exp(-r*T) * sum(payoffs)/runs

print("European exponential put option price (using Monte Carlo integration) is: ")
print(round(Price6,2))
