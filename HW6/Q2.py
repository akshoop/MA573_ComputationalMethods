#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW6, question 2)

"""

# importing the necessary packages
import numpy as np
from numpy import sqrt, exp
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
y = -1
lam = 5
kappa = -1.5
rho = -0.2
xi = 0.25
r = 0.03
T = 0.3
runs = 1000

# payoff function for Lookback Put option
def payoffLP(Smax, ST):
    return max(Smax - ST, 0)

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
    while temp < 500:
        Y = np.append(Y, Y[temp] + lam*(kappa - Y[temp])*(T/500.) + xi*(W2[temp+1] - W2[temp]))
        temp += 1
        
    # St simulation (MILSTEIN SCHEME)
    # derivative of exp(Yt)*St is just exp(Yt)
    temp = 0
    St = np.zeros(0)
    St = np.append(St, s)
    while temp < 500:
        St = np.append(St, St[temp] 
                       + r*St[temp]*(T/500.) 
                       + exp(Y[temp])*St[temp]*(W1[temp+1] - W1[temp]) 
                       + (0.5 * exp(Y[temp])*exp(Y[temp])*St[temp]*((W1[temp+1] - W1[temp])**2 - (T/500.))))
        temp += 1
    
    # getting Smax from simulated St's, and ST terminal value
    Smax = max(St)
    ST = St[St.size - 1]
    
    # calculating payoff
    payoffsArray = np.append(payoffsArray, payoffLP(Smax, ST))
    
    num += 1
# calculating price
PriceLP = exp(-r*T)*np.mean(payoffsArray)
print("The price of the lookback put option is: ")
print(round(PriceLP,2))
