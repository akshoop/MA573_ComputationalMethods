#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Akex Shoop (akshoop)
HW 9, Question 3

"""

# importing the necessary packages
import numpy as np
from numpy import sqrt, exp, log
from scipy.stats import norm
import triSolver

##############################################################################

# initial vars
S0 = 100.
sigma = 0.20
r = 0.03
T = 1.
K = 115.

# on interval [0, 200]
# assume Smax = 2*S0
Smax = 200.
ds = Smax/1000.      # space step
M = 1000
timestep = 20.  # change this according to question part
dt = T/timestep  # time step
N = int(T/dt)

# stock steps
s = np.array([ds*j for j in range(M)])

# initial v setup
v = np.maximum(s - K, 0)

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

# setting up LHS and RHS matrices for price equation
leftB = I - 0.5*dt*A
rightB = I + 0.5*dt*A

# Crank-Nicolson scheme
for k in range(N):
    b_tilda = np.dot(rightB, v)
    v = triSolver.betterSolver(leftB, b_tilda)

# closed form BS formula price
d1 = ( log(S0/K) + (r + (sigma**2)/2)*T ) / ( sigma*sqrt(T) )
d2 = d1 - sigma*sqrt(T)
BSCallPrice = S0*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
print("Call Price explicitly via BS model formula = " + str(BSCallPrice))
print("-------------------------------------------------")    

print("Crank-Nicolson finite difference method estimated price (for timestep = " + str(timestep) + ") is:")
print(v[(M+1)/2])


