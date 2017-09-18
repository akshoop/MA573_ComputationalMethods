#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW5, question 3)

"""

# importing the necessary packages
import numpy as np
from numpy import exp, log, sqrt
import matplotlib.pyplot as plt

##############################################################################

# Ornstein-Uhlenbeck (OU) process

# initial vars
lam = 2.
kappa = 120.
sigma = 25.
X_0 = 100.

N = 1000.
T = 1
dt = T/N

# t variable for x-axis plotting
t = np.arange(0, 1, 0.001)

num = 0
while num < 10:
    X_t = np.array([X_0])
    
    i = 0
    while (i+1) < N:
        Z = np.random.standard_normal()
        X_t = np.append(X_t, 
                        X_t[i] + lam*(kappa - X_t[i])*dt + sigma*sqrt(dt)*Z)
        i += 1
    
    plt.plot(t, X_t)
    num += 1

plt.title("Ornstein-Uhlenbeck (OU) process")
plt.xlabel("Time t")
plt.ylabel("Process value")

plt.show()

####################################
# changed parameter plots

lam_edit = 10.
kappa_edit = 200.
sigma_edit = 10.

plt.subplot(2,2,1)
num = 0
while num < 10:
    X_t = np.array([X_0])
    
    i = 0
    while (i+1) < N:
        Z = np.random.standard_normal()
        X_t = np.append(X_t, 
                        X_t[i] + lam_edit*(150 - X_t[i])*dt + sigma*sqrt(dt)*Z)
        i += 1
    
    plt.plot(t, X_t)
    num += 1

plt.title("Modified Lambda (Lambda = 10)")
plt.xlabel("Time t")
plt.ylabel("Process value")

plt.subplot(2,2,2)
num = 0
while num < 10:
    X_t = np.array([X_0])
    
    i = 0
    while (i+1) < N:
        Z = np.random.standard_normal()
        X_t = np.append(X_t, 
                        X_t[i] + lam*(kappa_edit - X_t[i])*dt + sigma*sqrt(dt)*Z)
        i += 1
    
    plt.plot(t, X_t)
    num += 1

plt.title("Modified Kappa (Kappa = 200)")
plt.xlabel("Time t")
plt.ylabel("Process value")

plt.subplot(2,2,3)
num = 0
while num < 10:
    X_t = np.array([X_0])
    
    i = 0
    while (i+1) < N:
        Z = np.random.standard_normal()
        X_t = np.append(X_t, 
                        X_t[i] + lam*(kappa - X_t[i])*dt + sigma_edit*sqrt(dt)*Z)
        i += 1
    
    plt.plot(t, X_t)
    num += 1

plt.title("Modified Sigma (Sigma = 10)")
plt.xlabel("Time t")
plt.ylabel("Process value")

plt.show()