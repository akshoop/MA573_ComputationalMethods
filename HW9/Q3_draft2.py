#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Akex Shoop (akshoop)
HW 9, Question 3

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
T = 1.
K = 115.

# on interval [0, 200]
# assume Smax = 2*S0
Smax = 200.
ds = Smax/1000.      # space step
M = 1000
timestep = 50e.  # change this according to question part
dt = T/timestep  # time step
N = int(T/dt)

# stock steps
s = np.array([ds*j for j in range(M)])

# initial v setup
v = np.maximum(s - K, 0)

# basic alpha, beta, gamma matrix A version
alpha = 0.5  * (sigma**2*s**2/ds**2 - r*s**2/ds)
beta = -sigma**2*s**2/ds**2 - r
gamma = 0.5  * (sigma**2*s**2/ds**2 + r*s**2/ds)

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

# invert
leftB_inv = np.linalg.inv(leftB)

# regular
regularA = I + dt*A
"""
# setting up leftB
leftB[0, :] = 0
leftB[M-1, :] = 0
leftB[M-1, -3:] = np.array([1,-2,1])

# invert
#leftB_inv = np.linalg.inv(leftB)

# boundary conditions
rightB[0, :] = 0
#leftB_inv[0, :] = 0

rightB[M-1, :] = 0
rightB[M-1, -3:] = np.array([1,-2,1])
#leftB_inv[M-1, :] = 0
#leftB_inv[M-1, -3:] = np.array([1,-2,1])
"""



# use same alpha, beta, gamma matrix A from Q2
# use regular matrix A, not the identity guy

# leftB*v_n+1 = rightB*v_n
# leftB = I - 0.5*dt*A
# rightB = I + 0.5*dt*A

# then do v_n+1 multiplication inverse stuff

# FIX TRI SOLVER
# try to make regular solver as lower triangular solver instead of upper triangular
# Ruojing said her upper triagnular solver gave nan for Q3
# but when she changed it to lower tiangular solver, it gave better result



"""
# this is (Identity + (1/2)dt*A) big matrix
alpha1 = 0.5 * 0.5 * dt * (sigma**2*j**2 - r*j)
beta1 = -0.5*sigma**2*j**2*dt - 0.5*r*dt
gamma1 = 0.5 * 0.5 * dt * (sigma**2*j**2 + r*j)
# setting up matrix A1 
A1 = np.diag(beta1) + np.diag(alpha1[1:], -1) + np.diag(gamma1[0:M-2], 1)

# this is (Identity - (1/2)dt*A) big matrix
alpha2 = -0.5 * 0.5 * dt * (sigma**2*j**2 - r*j)
beta2 = 0.5*sigma**2*j**2*dt + 0.5*r*dt
gamma2 = -0.5 * 0.5 * dt * (sigma**2*j**2 + r*j)
# setting up matrix A_tilda
A_tilda = np.diag(beta2) + np.diag(alpha2[1:], -1) + np.diag(gamma2[0:M-2], 1)

# invert the A_tilda matrix for Crank-Nicolsan scheme calculation
Ainv = np.linalg.inv(A_tilda)

# boundary conditions for matrix A1 and Ainvert
A1[0, :] = 0     # initial is 0
Ainv[0, :] = 0

A1[M-2, :] = 0   # linearity boundary
A1[M-2, -3:] = np.array([1,-2,1])
Ainv[M-2, :] = 0
Ainv[M-2, -3:] = np.array([1,-2,1])

# initial V setup
stock_steps = np.arange(0, Smax+ds, ds)
v0 = np.array(np.maximum(stock_steps - K, 0))
v0 = v0[2:]     # reducing size for correct dimensions
v0.shape = (999,1)
v = np.zeros_like(v0)


# CRANK-NICOLSAN SCHEME
for i in range(0, M-1):
    
    if i == 0:
        b_tilda = np.dot(rightB, v)
        v = triSolver.generalizedSolver(leftB, b_tilda)
        #v = np.dot(leftB_inv, b_tilda)
    else:
        b_tilda = np.dot(rightB, v)
        v = triSolver.generalizedSolver(leftB, b_tilda)
        #v = np.dot(leftB_inv, b_tilda)
        #v = np.maximum(v, v0)   # compare with v0 payoff vector
"""

# CRANK-NICOLSON SCHEME
for k in range(int(timestep)):
    #v = np.dot(regularA, v)
    #v[0] = 0
    #v[-1] = (Smax - K)*np.exp((k+1)/timestep) 
    b_tilda = np.dot(rightB, v)
    #v = np.dot(leftB_inv, v)
    v = triSolver.tridiagonalSolver(leftB, b_tilda)
    v[0] = 0
    v[-1] = (Smax - K)*np.exp((k+1)/timestep)
    
print("Crank-Nicolson finite difference method estimated price (for timestep = " + str(timestep) + ") is:")
#print(v[(M+1)/2])
print(v)

# closed form BS formula price
d1 = ( log(S0/K) + (r + (sigma**2)/2)*T ) / ( sigma*sqrt(T) )
d2 = d1 - sigma*sqrt(T)
BSCallPrice = S0*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
print("Call Price explicitly via BS model formula = " + str(round(BSCallPrice,2)))
print("-------------------------------------------------")
