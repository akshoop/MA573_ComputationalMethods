#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW6, question 4)

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
y = 0.08
lam = 3
kappa = 0.1
rho = -0.8
xi = 0.1
r = 0.03

T = 1.0
runs = 1000
K = 85

# payoff function for European Call option
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

# big loop to get difference price values for different K values
i = 0
while i < 9:
    # array holding payoffs for price calculation
    payoffsArray = np.zeros(0)
    
    num = 0
    while num < runs:
        # get W1 and W~ Brownian Motions, in order to get the W2 BM
        # note: need to use rho correlation equation
        W1 = SimBMStep(T, 500.)
        Wtilda = SimBMStep(T, 500.)
        W2 = rho*W1 + sqrt(1 - rho**2)*Wtilda
        
        # simulating Yt values
        temp = 0
        Y = np.zeros(0)
        # getting initial Y_0, which is little y = -1
        Y = np.append(Y, y)
        # loop for Yt simulation
        # NOTE, Yt follows CIR process form
        while temp < 500:
            Y = np.append(Y, max(Y[temp] 
                                 + lam*(kappa - Y[temp])*(T/500.) 
                                 + xi*sqrt(Y[temp])*(W2[temp+1] - W2[temp]), 0))
            temp += 1
            
        # St simulation (MILSTEIN SCHEME)
        temp = 0
        St = np.zeros(0)
        St = np.append(St, s)
        while temp < 500:
            St = np.append(St, St[temp] 
                           + r*St[temp]*(T/500.) 
                           + sqrt(Y[temp])*St[temp]*(W1[temp+1] - W1[temp]) 
                           + (0.5 * sqrt(Y[temp])*sqrt(Y[temp])*St[temp]*((W1[temp+1] - W1[temp])**2 - (T/500.))))
            temp += 1
        
        ST = St[St.size - 1]
        # calculating payoff
        payoffsArray = np.append(payoffsArray, payoffEC(ST, K))
        
        num += 1
    # calculating price
    PriceECmarket = exp(-r*T)*np.mean(payoffsArray)
    print("K = " + str(K) + ", European Call price = " + str(round(PriceECmarket,2)))
    # assume volatility is 1.0, in order to get correct implied volatility
    ImpliedVolatility = ImplVol(PriceECmarket, 1.0)
    print("Implied volatility = " + str(ImpliedVolatility))
    
    K += 5
    i += 1
