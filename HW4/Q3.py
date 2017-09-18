#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW4, question 3)

"""

# importing the necessary packages
import numpy as np
from numpy import sqrt, exp, log
from scipy.stats import norm
import matplotlib.pyplot as plt

##############################################################################

# variables given
S_0 = 100.
B0 = 1
mu = 0.3
sigma = 0.3
r = 0.03

# Asian Call
K = 120.
T = 5

# number of simulation runs/sample size for Monte Carlo
runs = 10000

# number of average time for Asian payoff calculation
N = 1260.

dt = T/N

#####################################
# part c)

# Using the Geometric Asian Average as the Control Variate

# Need actual value of Geometric Asian call option
sigma_geo = sigma * sqrt(1/3.)
r_geo = 0.5 * (r - (1/6.) * sigma**2)
d1 = (log(S_0/K) + 0.5 * (r + (1/6.)*sigma**2) * T) / (sigma_geo * sqrt(T))
d2 = d1 - (sigma_geo * sqrt(T))
Asian_geo_price = exp(-(r*T)) * (S_0*exp(r_geo*T)*norm.cdf(d1) - K*norm.cdf(d2))

# initialize array to hold average values
Asian_avg_geo = np.zeros(0)
Asian_avg_arith = np.zeros(0)

# loop for getting St arithmetic average values. 
num = 0
while num < runs:
    BM_sum = 0
    S_t = np.zeros(0)

    # calculating St array
    i = 0    
    while i < int(N):
        # have to make new Brownian Motion for each increment step
        Z = np.random.standard_normal()
        BM_increment = sqrt(dt) * Z
        BM_sum += BM_increment
        
        S_t = np.append(S_t, S_0 * exp((r - (sigma**2)/2)*(i+1)*(dt) + sigma * BM_sum))
        i += 1
        
    # calculate average value
    Asian_avg_geo = np.append(Asian_avg_geo, np.product(S_t**(1/N)))
    Asian_avg_arith = np.append(Asian_avg_arith, sum(S_t)/S_t.size)
    num += 1
    
# Asian geometric call payoffs
temp = Asian_avg_geo - K
for x in np.nditer(temp, op_flags = ['readwrite']):
    x[...] = max(x, 0) # in order to make negative values into 0
Asian_call_payoffs_geo = temp

# Asian arithmatic call payoffs
temp = Asian_avg_arith - K
for x in np.nditer(temp, op_flags = ['readwrite']):
    x[...] = max(x, 0) # in order to make negative values into 0
Asian_call_payoffs_arith = temp

# calculate the optimal b value for the equation sampling
Cov_test = np.cov(Asian_call_payoffs_geo, Asian_call_payoffs_arith)
b_optimal = Cov_test[0,1] / Cov_test[1,1]

# new sampling, using the control variate
Asian_sampling = Asian_call_payoffs_arith - b_optimal * (Asian_call_payoffs_geo - Asian_geo_price)

Asian_call_price = exp(-r*T) * np.mean(Asian_sampling)
print("The estimated price of arithmetic average Asian Call Option (with Control Variate) is:")
print(round(Asian_call_price,2))

#####################################
# part d)

Y_asian_call_control = np.zeros(0)
Y_asian_call_NoControl = np.zeros(0)

# getting the respective arithmetic Asian call prices (WITH CONTROL VARIATE)
# Asian_sampling is from part c)
for i in range(runs):
    Y_asian_call_control = np.append(Y_asian_call_control,
                                       exp(-r*T) * (sum(Asian_sampling[:(i+1)]) / (Asian_sampling[:(i+1)].size)))

# getting the respective arithmetic Asian call prices based on sample size
for j in range(runs):
    Y_asian_call_NoControl = np.append(Y_asian_call_NoControl,
                                       exp(-r*T) * 
                                        (sum(Asian_call_payoffs_arith[:(j+1)]) 
                                        / (Asian_call_payoffs_arith[:(j+1)].size)))
 

plt.plot(np.arange(runs), Y_asian_call_control, label = "Monte Carlo w/ Control Variate")
plt.plot(np.arange(runs), Y_asian_call_NoControl, label = "Direct Monte Carlo")

plt.title("Monte Carlo Calculation of Arithmetic Asian Call Price")
plt.xlabel("Sample size")
plt.ylabel("Price")

plt.legend(loc = 1)
plt.show()

#####################################
# part d)
print("----------------------------------------------------------------------")
print("Actual price of Asian geometric call is: ")
print(round(Asian_geo_price,2))

print("The estimated price of arithmetic average Asian Call Option (with Control Variate) is:")
print(round(Asian_call_price,2))
print("The estimated price of arithmetic average Asian Call (using only Direct Monte Carlo) is: ")    
print(round(Y_asian_call_NoControl[runs - 1], 2))

# calculations for Euro Call calculation

Z1 = np.random.standard_normal(runs)
# calculating S_T array
S_T = S_0 * exp((r - (sigma**2)/2)*T + sigma * sqrt(T) * Z1)
# calculation of European call payoffs
temp = S_T - K
for x in np.nditer(temp, op_flags = ['readwrite']):
    x[...] = max(x, 0) # in order to make negative values into 0
Euro_call_payoffs = temp
Euro_call_price = exp(-r*T) * np.mean(Euro_call_payoffs)

print("Price of European call option is: ")
print(round(Euro_call_price,2))
