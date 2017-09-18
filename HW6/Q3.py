#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW6, question 3)

"""

# importing the necessary packages
import numpy as np
from numpy import sqrt, exp, log
from scipy.stats import norm
import matplotlib.pyplot as plt

##############################################################################

# increment step method for Brownian Motion
def SimBMStep(T, N):
    # initialize W brownian motion array
    W = np.zeros(int(N))
    W = np.append(W, 0)    # initial 0 value
    
    # loop
    num = 0
    while num < N:
        W[num + 1] = W[num] + sqrt(T/N)*np.random.standard_normal()
        num += 1
        
    return W
    
# initial vars
s = 100.
sigma = 0.2
r = 0.03
T = 1.
runs = 1000
K = 85
beta = 0
print("Beta is " + str(beta))

# payoff function for European call option
def payoffEC(ST, K):
    return max(ST - K, 0)

# defined function to get implied volatility
def ImplVol(MarketPrice, sigma_test):
    while sigma_test < 2.0:
        d1 = (log(s/K) + (r + (sigma_test**2)/2)*T) / (sigma_test * sqrt(T))
        d2 = d1 - sigma_test*sqrt(T)       
        # black scholes formula price calculation
        PriceBS = s*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
        # check if the two prices are close to each other
        if abs(PriceBS - MarketPrice) < 0.001:
            return sigma_test
        # else, do recursion of volatility calculation
        else:
            vega = s*sqrt(T)*norm.pdf(d1)
            sigma_test = sigma_test - (PriceBS - MarketPrice)/vega

# big loop to print out various K values
i = 0
while i < 9:    
    
    # array holding payoffs for price calculation
    payoffsArray = np.zeros(0)
    
    num = 0
    while num < runs:
        # get W1 BM
        W1 = SimBMStep(T, 500.)
            
        # St simulation
        temp = 0
        St = np.zeros(0)
        St = np.append(St, s)
        while temp < 500:
            St = np.append(St, St[temp] 
                           + r*St[temp]*(T/500.) 
                           + sigma*(St[temp]**beta)*St[temp]*(W1[temp+1] - W1[temp]))
            temp += 1
        # terminal ST
        ST = St[St.size - 1]
        # calculating payoff
        payoffsArray = np.append(payoffsArray, payoffEC(ST, K))
        num += 1
        
    # calculating price
    PriceECmarket = exp(-r*T)*np.mean(payoffsArray)
    print("K = " + str(K) + ", beta = " + str(beta) + ", Euro Call Price = " + str(round(PriceECmarket,2)))
    
    # assume volatility is 1.0, in order to get correct implied volatility
    ImpliedVolatility = ImplVol(PriceECmarket, 1.0)
    print("Implied volatility = " + str(ImpliedVolatility))
    
    
    K += 5
    i += 1

