#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW4, question 2)

"""

# importing the necessary packages
import numpy as np
from numpy import sqrt, exp, log
import matplotlib.pyplot as plt

##############################################################################

plt.clf()

# variables given
S_0 = 100.
B0 = 1
mu = 0.1
sigma = 0.2
r = 0.02

# number of simulation runs/sample size for Monte Carlo
runs = 5000

# initalize our Z array used for S_T calculation
Z1 = np.random.standard_normal(runs)

###########################################
# part a)

plt.subplot(2,3,1)

T = 0.5
K = 95.
N = 500     
#dt = T/runs

# calculating S_T array
S_T = S_0 * exp((r - (sigma**2)/2)*T + sigma * sqrt(T) * Z1)

# calculation of European put payoffs
temp = K - S_T
for x in np.nditer(temp, op_flags = ['readwrite']):
    x[...] = max(x, 0) # in order to make negative values into 0
Euro_put_payoffs1 = temp

# calculation of correlation
Correlation_a = np.corrcoef(S_T, Euro_put_payoffs1)
print("The correlation between underlying stock and a European put option is: ")
print(Correlation_a[0,1])

# plotting
plt.scatter(S_T, Euro_put_payoffs1)
plt.title("Underlying & Euro Put")
plt.xlabel("Underlying Stock")
plt.ylabel("Payoffs")

###########################################
# part b)

plt.subplot(2,3,2)

# use same T and K

# calculating S_T array
S_T = S_0 * exp((r - (sigma**2)/2)*T + sigma * sqrt(T) * Z1)

# calculation of European call payoffs
temp = S_T - K
for x in np.nditer(temp, op_flags = ['readwrite']):
    x[...] = max(x, 0) # in order to make negative values into 0
Euro_call_payoffs = temp

# calculation of correlation
Correlation_b = np.corrcoef(S_T, Euro_call_payoffs)
print("The correlation between underlying stock and a European call option is: ")
print(Correlation_b[0,1])

# plotting
plt.scatter(S_T, Euro_call_payoffs)
plt.title("Underlying & Euro Call")
plt.xlabel("Underlying Stock")
plt.ylabel("Payoffs")

###########################################
# part c)

plt.subplot(2,3,3)

# use same T and K

# initialize array to hold average values
Asian_average_arith1 = np.zeros(0)

# array to hold S(T) terminal values, for plotting purposes
ST_array1 = np.zeros(0)
    
# loop for getting St arithmetic average values. Always use (1/N) for
# average calculation

# USING 180 DAYS AS HALF A YEAR (T = 0.5)

num = 0
while num < runs:
    BM_sum = 0
    S_t = np.zeros(0)
    
    # consider cumulative sum if time permits
    
    # calculating St array
    i = 0    
    while i < 180:
        # have to make new Brownian Motion for each increment step
        Z2 = np.random.standard_normal()
        BM_increment = sqrt(1/180.) * Z2
        BM_sum += BM_increment
        
        S_t = np.append(S_t, S_0 * exp((r - (sigma**2)/2)*(i+1)*(1/180.) + sigma * BM_sum))
        i += 1
        
    # keep the S_T terminal value for plotting
    ST_array1 = np.append(ST_array1, S_t[S_t.size - 1])
    
    # calculate average value
    Asian_average_arith1 = np.append(Asian_average_arith1, sum(S_t)/S_t.size)
    num += 1
    
# Asian arithmatic put payoffs
temp = K - Asian_average_arith1
for x in np.nditer(temp, op_flags = ['readwrite']):
    x[...] = max(x, 0) # in order to make negative values into 0
Asian_put_payoffs_arith1 = temp


# calculation of correlation
Correlation_c = np.corrcoef(ST_array1, Asian_put_payoffs_arith1)
print("The correlation between underlying stock and arithmetic average Asian put is:  ")
print(Correlation_c[0,1])

# plotting
plt.scatter(ST_array1, Asian_put_payoffs_arith1)
plt.title("Underlying & arithmetic average Asian Put")
plt.xlabel("Underlying Stock")
plt.ylabel("Payoffs")


###########################################
# part d)

plt.subplot(2,3,4)

# use same T and K

# initialize array to hold average values
Asian_average_arith2 = np.zeros(0)

# array to hold S(T) terminal values, for Euro put payoff calculations
ST_array2 = np.zeros(0)

# loop for getting St arithmetic average values.
num = 0
while num < runs:
    BM_sum = 0
    S_t = np.zeros(0)
    
    # consider cumulative sum if time permits
    
    # calculating St array
    i = 0    
    while i < 180:
        # have to make new Brownian Motion for each increment step
        Z2 = np.random.standard_normal()
        BM_increment = sqrt(1/180.) * Z2
        BM_sum += BM_increment
        
        S_t = np.append(S_t, S_0 * exp((r - (sigma**2)/2)*(i+1)*(1/180.) + sigma * BM_sum))
        i += 1
    
    # keep the S_T terminal value for Euro put payoff calculation
    ST_array2 = np.append(ST_array2, S_t[S_t.size - 1])
        
    # calculate average value
    Asian_average_arith2 = np.append(Asian_average_arith2, sum(S_t)/S_t.size)
    num += 1
    
# Asian arithmatic put payoffs
temp = K - Asian_average_arith2
for x in np.nditer(temp, op_flags = ['readwrite']):
    x[...] = max(x, 0) # in order to make negative values into 0
Asian_put_payoffs_arith2 = temp

# calculation of European put payoffs
temp = K - ST_array2
for x in np.nditer(temp, op_flags = ['readwrite']):
    x[...] = max(x, 0) # in order to make negative values into 0
Euro_put_payoffs2 = temp

# calculation of correlation
Correlation_d = np.corrcoef(Euro_put_payoffs2, Asian_put_payoffs_arith2)
print("The correlation between European put and arithmetic Asian put is: ")
print(Correlation_d[0,1])

# plotting
plt.scatter(Euro_put_payoffs2, Asian_put_payoffs_arith2)
plt.title("Euro Put & arithmetic average Asian Put")
plt.xlabel("Euro payoffs")
plt.ylabel("Asian payoffs")

###########################################
# part e)

plt.subplot(2,3,5)

# use same T and K

# initialize array to hold average values
Asian_average_geo1 = np.zeros(0)

# array to hold S(T) terminal values, for Euro put payoff calculations
ST_array3 = np.zeros(0)

# loop for getting St arithmetic average values.
num = 0
while num < runs:
    BM_sum = 0
    S_t = np.zeros(0)
    
    # consider cumulative sum if time permits
    
    # calculating St array
    i = 0    
    while i < 180:
        # have to make new Brownian Motion for each increment step
        Z2 = np.random.standard_normal()
        BM_increment = sqrt(1/180.) * Z2
        BM_sum += BM_increment
        
        S_t = np.append(S_t, S_0 * exp((r - (sigma**2)/2)*(i+1)*(1/180.) + sigma * BM_sum))
        i += 1
    
    # keep the S_T terminal value for Euro put payoff calculation
    ST_array3 = np.append(ST_array3, S_t[S_t.size - 1])
        
    # calculate average value
    # slight adjustment of product and nth root calculation
    # first take nth root, then take product
    Asian_average_geo1 = np.append(Asian_average_geo1, np.product(S_t**(1/180.)))
    num += 1
    
# Asian geometric put payoffs
temp = K - Asian_average_geo1
for x in np.nditer(temp, op_flags = ['readwrite']):
    x[...] = max(x, 0) # in order to make negative values into 0
Asian_put_payoffs_geo1 = temp

# calculation of European put payoffs
temp = K - ST_array3
for x in np.nditer(temp, op_flags = ['readwrite']):
    x[...] = max(x, 0) # in order to make negative values into 0
Euro_put_payoffs3 = temp

# calculation of correlation
Correlation_e = np.corrcoef(Euro_put_payoffs3, Asian_put_payoffs_geo1)
print("The correlation between European put and geometric Asian put is: ")
print(Correlation_e[0,1])

# plotting
plt.scatter(Euro_put_payoffs3, Asian_put_payoffs_geo1)
plt.title("Euro Put & geometric average Asian Put")
plt.xlabel("Euro payoffs")
plt.ylabel("Asian payoffs")

###########################################
# part f)

plt.subplot(2,3,6)

# use same T and K

# initialize array to hold average values
Asian_average_geo2 = np.zeros(0)
Asian_average_arith3 = np.zeros(0)

# loop for getting St arithmetic average values. 
num = 0
while num < runs:
    BM_sum = 0
    S_t = np.zeros(0)
    
    # consider cumulative sum if time permits
    
    # calculating St array
    i = 0    
    while i < 180:
        # have to make new Brownian Motion for each increment step
        Z2 = np.random.standard_normal()
        BM_increment = sqrt(1/180.) * Z2
        BM_sum += BM_increment
        
        S_t = np.append(S_t, S_0 * exp((r - (sigma**2)/2)*(i+1)*(1/180.) + sigma * BM_sum))
        i += 1
    
    # keep the S_T terminal value for Euro put payoff calculation
    #ST_array4 = np.append(ST_array2, S_t[S_t.size - 1])
        
    # calculate average value
    Asian_average_geo2 = np.append(Asian_average_geo2, 
                                   np.product(S_t**(1/180.)))
    Asian_average_arith3 = np.append(Asian_average_arith3, sum(S_t)/S_t.size)
    num += 1
    
# Asian geometric put payoffs
temp = K - Asian_average_geo2
for x in np.nditer(temp, op_flags = ['readwrite']):
    x[...] = max(x, 0) # in order to make negative values into 0
Asian_put_payoffs_geo2 = temp

# Asian arithmatic put payoffs
temp = K - Asian_average_arith3
for x in np.nditer(temp, op_flags = ['readwrite']):
    x[...] = max(x, 0) # in order to make negative values into 0
Asian_put_payoffs_arith3 = temp

# calculation of correlation
Correlation_f = np.corrcoef(Asian_put_payoffs_geo2, Asian_put_payoffs_arith3)
print("The correlation between European put and geometric Asian put is: ")
print(Correlation_f[0,1])

# plotting
plt.scatter(Asian_put_payoffs_geo2, Asian_put_payoffs_arith3)
plt.title("Geometric Asian put & arithmetic Asian Put")
plt.xlabel("Geometric payoffs")
plt.ylabel("Arithmetic payoffs")

plt.show()


