#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Akex Shoop (akshoop)
HW 9, Question 2

"""

# importing the necessary packages
import numpy as np
from numpy import sqrt, exp, log, mean
from scipy.stats import norm
import triSolver

##############################################################################

# initial vars
S0 = 100.
sigma = 0.20
r = 0.03
T = 1
K = 115.

# on interval [0, 200]
# assume Smax = 2*S0
Smax = 200.
ds = Smax/1000.      # space step
M = 1000
timestep = 20.
dt = T/timestep  # time step
N = int(T/dt)

# getting necessary coefficients from PDE
j = np.arange(1, M)

# this is (Identity + dt*A) version
alpha = 0.5 * dt * (sigma**2*j**2 - r*j)
beta = 1-sigma**2*j**2*dt - r*dt
gamma = 0.5 * dt * (sigma**2*j**2 + r*j)

# setting up matrix A 
A = np.diag(beta) + np.diag(alpha[1:], -1) + np.diag(gamma[0:M-2], 1)

# boundary conditions for matrix A
A[0, :] = 0     # initial is 0
A[M-2, :] = 0   # linearity boundary
A[M-2, -3:] = np.array([1,-2,1])

# initial V setup
stock_steps = np.arange(0, Smax+ds, ds)
v0 = np.array(np.maximum(stock_steps - K, 0))
v0 = v0[2:]     # reducing size for correct dimensions
v0.shape = (999,1)
v = np.zeros_like(v0)

# EXPLICIT SCHEME
for i in range(0, M-1):
    if i == 0:
        v = np.dot(A, v0)
    else:
        v = np.dot(A, v)

print(v)
print("Explicit finite difference method estimated price (for timestep = " + str(timestep) + ") is:")
print(v[(M+1)/2])


# closed form BS formula price
d1 = ( log(S0/K) + (r + (sigma**2)/2)*T ) / ( sigma*sqrt(T) )
d2 = d1 - sigma*sqrt(T)
BSCallPrice = S0*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
print("Call Price explicitly via BS model formula = " + str(round(BSCallPrice,2)))
print("-------------------------------------------------")

x = triSolver.tridiagonalSolver(Atemp, Btemp)
