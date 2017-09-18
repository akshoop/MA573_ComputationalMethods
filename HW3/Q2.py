#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Alex Shoop (akshoop)

HW3, question 2)

"""

# importing the necessary packages
import numpy as np

##############################################################################

alpha = 0.6
#lambda = 1

losses = np.zeros(0)

# loss random variable equation
def loss(y):
    return 100000*y - 220000

n = 0
while n < 100000:
    x = loss(np.random.weibull(alpha))
    losses = np.append(losses, x)
    n += 1

# sort losses array from largest to lowest
losses = np.sort(losses)
losses[:] = losses[::-1]

# select the loss (95% value at risk) that is the specified one
# 100000*0.05 = 5000

print("Numerical estimation of 95% Value At Risk is:")
print(losses[5000])


