#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW1, question 1
"""

# importing the necessary packages
import numpy as np
import matplotlib.pyplot as plt


##############################################################################
plt.subplot(2, 2, 1)

unif10 = np.random.uniform(low = 0, high = 1, size = 10)
print("The samples of size 10 are: ")
print(unif10)

# empirical CDF
H, X1 = np.histogram(unif10, bins = 10)
# the following two lines are necessary to get the precise information
# for getting the empirical values.
dX = X1[1] - X1[0]
empCDF = np.cumsum(H)*dX
plt.plot(X1[1:], empCDF, color = 'b', label = "Empirical CDF")

# actual CDF
X2 = np.sort(unif10)
unif10 /= unif10.sum()
actualCDF = np.cumsum(unif10)
plt.plot(X2, actualCDF, color = 'r', label = "Actual CDF")

plt.title("Empirical CDF vs Actual CDF (sample size 10)")
plt.legend()

##############################################################################
plt.subplot(2, 2, 2)

unif100 = np.random.uniform(low = 0, high = 1, size = 100)
print("The samples of size 100 are: ")
print(unif100)

# empirical CDF
H, X1 = np.histogram(unif100, bins = 100)
dX = X1[1] - X1[0]
empCDF = np.cumsum(H)*dX
plt.plot(X1[1:], empCDF, color = 'b', label = "Empirical CDF")

# actual CDF
X2 = np.sort(unif100)
unif100 /= unif100.sum()
actualCDF = np.cumsum(unif100)
plt.plot(X2, actualCDF, color = 'r', label = "Actual CDF")

plt.title("Empirical CDF vs Actual CDF (sample size 100)")
plt.legend()

##############################################################################
# for the samples of size 1000 and beyond, I decided to print only the plots
plt.subplot(2, 2, 3)

unif1000 = np.random.uniform(low = 0, high = 1, size = 1000)

# empirical CDF
H, X1 = np.histogram(unif1000, bins = 1000)
dX = X1[1] - X1[0]
empCDF = np.cumsum(H)*dX
plt.plot(X1[1:], empCDF, color = 'b', label = "Empirical CDF")

# actual CDF
X2 = np.sort(unif1000)
unif1000 /= unif1000.sum()
actualCDF = np.cumsum(unif1000)
plt.plot(X2, actualCDF, color = 'r', label = "Actual CDF")

plt.title("Empirical CDF vs Actual CDF (sample size 1000)")
plt.legend()

##############################################################################
plt.subplot(2, 2, 4)

unif10000 = np.random.uniform(low = 0, high = 1, size = 10000)

# empirical CDF
H, X1 = np.histogram(unif10000, bins = 10000)
dX = X1[1] - X1[0]
empCDF = np.cumsum(H)*dX
plt.plot(X1[1:], empCDF, color = 'b', label = "Empirical CDF")

# actual CDF
X2 = np.sort(unif10000)
unif10000 /= unif10000.sum()
actualCDF = np.cumsum(unif10000)
plt.plot(X2, actualCDF, color = 'r', label = "Actual CDF")

plt.title("Empirical CDF vs Actual CDF (sample size 10000)")
plt.legend()

##############################################################################

plt.tight_layout()

plt.show()