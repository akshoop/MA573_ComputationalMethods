#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW1, question 3

I decided to plot using 5000 samples from each RNG (uniform and LCRNG)

"""

# importing the necessary packages
import numpy as np
import matplotlib.pyplot as plt


##############################################################################
plt.subplot(2, 2, 1)

# setting the specified seed for uniform RNG
np.random.seed(375)
X1 = np.random.uniform(low = 0, high = 1, size = 1)

plt.scatter(X1, X1)
plt.title("Correlation using Uniform RNG (size = 1, seed = 375)")


##############################################################################
plt.subplot(2, 2, 2)

# defining LCRNG function
def LCRNG(val):
    newval = (a * val + c) % m
    return newval

# setting up variables
m = 11
a = 6
c = 0
seed = 1

# starting X2 array, which we will populate
X2 = np.zeros(5000)
X2_size = X2.size

# setting up first value, which is 'seed' mod m
X2[0] = seed % m

# looping through X2 array, running LCRNG function on each value
# X2_enum gives us the actual element at the i index
for i, X2_enum in enumerate(X2[:X2_size - 1]):
    X2[i + 1] = LCRNG(X2_enum)

# now creating U array, as detailed from lecture 1
U2 = X2
# np.nditer is another form of iterating across the numpy array
# we specifcy the flag 'readwrite' in order to overwrite the index i element
for i in np.nditer(U2, op_flags = ['readwrite']):
    i[...] = i / m

plt.scatter(U2[:X2_size - 1], U2[1:])
plt.title("Correlation using LCRNG (size = 5000, m = 11, a = 6, c = 0)")

##############################################################################
plt.subplot(2, 2, 3)

# setting up variables
m = 2 ** 31 - 1
a = 16807
c = 0
seed = 1

# starting X3 array, which we will populate
X3 = np.zeros(5000)
X3_size = X3.size

# setting up first value, which is 'seed' mod m
X3[0] = seed % m

# looping through X2 array, running LCRNG function on each value
# X3_enum gives us the actual element at the i index
for i, X3_enum in enumerate(X3[:X3_size - 1]):
    X3[i + 1] = LCRNG(X3_enum)

# now creating U array, as detailed from lecture 1
U3 = X3
for i in np.nditer(U3, op_flags = ['readwrite']):
    i[...] = i / m

plt.scatter(U3[:X3_size - 1], U3[1:])
plt.title("Correlation using LCRNG (size = 5000, m = 2^31 - 1, a = 16807, c = 0)")

##############################################################################
plt.subplot(2, 2, 4)

# setting up variables
m = 2 ** 31 - 1
a = 950706376
c = 0
seed = 1

# starting X4 array, which we will populate
X4 = np.zeros(5000)
X4_size = X4.size

# setting up first value, which is 'seed' mod m
X4[0] = seed % m

# looping through X2 array, running LCRNG function on each value
# X4_enum gives us the actual element at the i index
for i, X4_enum in enumerate(X4[:X4_size - 1]):
    X4[i + 1] = LCRNG(X4_enum)

# now creating U array, as detailed from lecture 1
U4 = X4
for i in np.nditer(U4, op_flags = ['readwrite']):
    i[...] = i / m

plt.scatter(U4[:X4_size - 1], U4[1:])
plt.title("Correlation using LCRNG (size = 5000, m = 2^31 - 1, a = 950706376, c = 0)")


plt.show()