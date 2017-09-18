#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW5, question 4)

"""

# importing the necessary packages
import numpy as np
from numpy import exp, log, sqrt
import matplotlib.pyplot as plt

##############################################################################
# interest rate swap

# initial vars
lam = 0.7
kappa = 0.05
sigma = 0.006
r_0 = 0.02
r_fix = 0.04
N_notional = 10000

N = 3000.
T = 3.
dt = T/N

runs = 10000

# interest rate follows Ornstein-Uhlenbeck dynamics
r_T = np.zeros(0)

num = 0
while num < runs:
    
    # loop for r_t process
    r_t = np.array([r_0])  
    i = 0
    
    while (i+1) < N:
        Z = np.random.standard_normal()
        r_t = np.append(r_t, r_t[i] + lam*(kappa - r_t[i])*dt + sigma*sqrt(dt)*Z)
        i += 1
        
    # integral r_s calculation
    r_T = np.append(r_T, sum(r_t*dt))
    
    num += 1

payoffs = N_notional * (exp(r_T) - exp(r_fix * T))

PriceIntSwap = np.mean(payoffs)

print("Expected payoff of the interest rate swap at maturity T = 2 is: ")
print(round(PriceIntSwap,2))
