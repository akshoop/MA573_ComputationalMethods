#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW1, question 4
"""

# importing the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import math #used for floor function

# FIRST LCRNG
##############################################################################

# setting up variables
m = 7
a = 5

# defining LCRNG function
def LCRNG(val, a, m):
    newval = (a * val) % m
    return newval

# starting X1 array, which we will populate
X1 = []

# setting up first value, and second value to get started on LCRNG iteration
X1.append(3)
X1.append(0)

# looping through X1 array, running LCRNG function on each value
# X1_enum gives us the actual element at the i index
# this loop also stops when X1 at index n is the same value as X1[0] = 3
for i, X1_enum in enumerate(X1):
    X1[i + 1] = LCRNG(X1_enum, a, m)
    
    if X1[i + 1] != X1[0]: 
        # append to the index, and iterate again
        X1.append(0)
    else:
        break

# now creating U array, as detailed from lecture 1
U1 = np.array(X1, dtype = 'float')
# np.nditer is another form of iterating across the numpy array
# we specifcy the flag 'readwrite' in order to overwrite the index i element
for j in np.nditer(U1, op_flags = ['readwrite']):
    j[...] = j / m

print("")
print("Our first LCRNG is: ")
print(U1)
print("And the period length = %d" % (len(X1) - 1))


# SECOND LCRNG
##############################################################################

# setting up variables
m = 5
a = 7

# starting X2 array, which we will populate
X2 = []

# setting up first value, and second value to get started on LCRNG iteration
X2.append(1)
X2.append(0)

# looping through X2 array, running LCRNG function on each value
for i, X2_enum in enumerate(X2):
    X2[i + 1] = LCRNG(X2_enum, a, m)
    
    if X2[i + 1] != X2[0]: 
        # append to the index, and iterate again
        X2.append(0)
    else:
        break

# now creating U array, as detailed from lecture 1
U2 = np.array(X2, dtype = 'float')
for j in np.nditer(U2, op_flags = ['readwrite']):
    j[...] = j / m

print("Our second LCRNG is: ")
print(U2)
print("And the period length = %d" % (len(X2) - 1))

# WICHMANN-HILL RNG
##############################################################################

U_WH = np.zeros_like(U1)

# need to add 0s towards end of U2, in order for U1 and U2 to be of same length
# for summation caluclation
U2_extended = np.append(U2, 2 * [0])

for k, UWH_enum in enumerate(U_WH):
    
    U_WH[k] = (U1[k] + U2_extended[k]) - math.floor(U1[k] + U2_extended[k])

print("Finally, our Wichmann-Hill RNG is:")
print(U_WH)
print("And period length = %d" % (len(U_WH) - 1))   


##############################################################################
##############################################################################

# plotting code

L1 = plt.scatter(U1[:U1.size - 1], U1[1:], color = 'b')
#plt.title("Correlation using LCRNG (m = 7, a = 5, x0 = 3)")

L2 = plt.scatter(U2[:U2.size - 1], U2[1:], color = 'r')
#plt.title("Correlation using LCRNG (m = 5, a = 7, x0 = 1)")

WH = plt.scatter(U_WH[:U_WH.size - 1], U_WH[1:], color = 'g')
plt.title("Serial Correlations")
plt.legend((L1, L2, WH), ('LCRNG1', 'LCRNG2', 'Wichmann-Hill RNG'))

plt.tight_layout()

plt.show()
