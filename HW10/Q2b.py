#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Akex Shoop (akshoop)
HW 10, Question 2

"""

# importing the necessary packages
import numpy as np
from numpy import sqrt, exp, log
from scipy.stats import norm
import triSolver

##############################################################################

# initial vars
S0 = 100.
sigma = 0.30
r = 0.02
T = 1.
K = 115.

# on interval [0, 200]
# assume Smax = 2*S0
Smax = 200.
ds = Smax/1000.      # space step
M = 1000
timestep = 100.  # change this according to question part
dt = T/timestep  # time step
N = int(T/dt)

# stock steps
s = np.array([ds*j for j in range(M)])

# initial v setup (put payoff)
v = np.maximum(K - s, 0)
v_tilda = v

# basic alpha, beta, gamma matrix A version
alpha = 0.5  * (sigma**2*s**2/ds**2 - r*s/ds)
beta = -sigma**2*s**2/ds**2 - r
gamma = 0.5  * (sigma**2*s**2/ds**2 + r*s/ds)

# setting up matrix A 
A = np.diag(beta) + np.diag(alpha[1:], -1) + np.diag(gamma[0:M-1], 1)

# boundary conditions for A
A[0, :] = 0
A[M-1, :] = 0
A[M-1, -3:] = np.array([1,-2,1])

# identity matrix
I = np.identity(1000)

# setting up main big matrix A_tilda for price equation
A_tilda = I - dt*A
#Ainv = np.linalg.inv(A_tilda)

# setting up matrix for Brennan-Schwartz algorithm
# ie. changing it to upper triangular
for i in range(0, len(A_tilda)-1):
    v_tilda[i+1] = v_tilda[i+1] - v_tilda[i]*A_tilda[i+1][i]/A_tilda[i][i]
    A_tilda[i+1] = A_tilda[i+1] - A_tilda[i]*A_tilda[i+1][i]/A_tilda[i][i]


# approximation of American Put option pricing calculation
for k in range(N):
    vtemp = triSolver.betterSolver(A_tilda, v_tilda)
    v = np.maximum(v, vtemp)
  

# closed form BS formula price
d1 = ( log(S0/K) + (r + (sigma**2)/2)*T ) / ( sigma*sqrt(T) )
d2 = d1 - sigma*sqrt(T)
BSCallPrice =  K*exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1) 
print("Put Price explicitly via BS model formula = " + str(BSCallPrice))
print("-------------------------------------------------")    

print("Implicit finite difference method estimated price (for timestep = " + str(timestep) + ") is:")
print(v[(M+1)/2])


