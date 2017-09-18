#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW1, question 2
"""

# importing the necessary packages
import numpy as np
import matplotlib.pyplot as plt


##############################################################################
plt.subplot(2, 2, 1)

unif10 = np.round(np.random.uniform(low = 0, high = 1, size = 10), 1)
print("The rounded samples of size 10 are: ")
print(unif10)

# necessary equation to make sure bin size is 0.1
binSize = (np.max(unif10) - np.min(unif10)) / 0.1

plt.hist(unif10, bins = binSize, normed = True)
plt.title("Relative frequency of samples size 10")

##############################################################################
plt.subplot(2, 2, 2)

unif100 = np.round(np.random.uniform(low = 0, high = 1, size = 100), 1)
print("The rounded samples of size 100 are: ")
print(unif100)

binSize = (np.max(unif100) - np.min(unif100)) / 0.1

plt.hist(unif100, bins = binSize, normed = True)
plt.title("Relative frequency of samples size 100")

##############################################################################
plt.subplot(2, 2, 3)


unif1000 = np.round(np.random.uniform(low = 0, high = 1, size = 1000), 1)

binSize = (np.max(unif1000) - np.min(unif1000)) / 0.1


plt.hist(unif1000, bins = binSize, normed = True)
plt.title("Relative frequency of samples size 1000")

##############################################################################
plt.subplot(2, 2, 4)


unif10000 = np.round(np.random.uniform(low = 0, high = 1, size = 10000), 1)

binSize = (np.max(unif10000) - np.min(unif10000)) / 0.1

plt.hist(unif10000, bins = binSize, normed = True)
plt.title("Relative frequency of samples size 10000")



plt.show()