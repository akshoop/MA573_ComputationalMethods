#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Akex Shoop (akshoop)
HW 9, Question 1

"""

# importing the necessary packages
import numpy as np
import scipy.linalg as linalg

##############################################################################

def tridiagonalSolver(A, b):
    # check if input matrix is of correct form
    if (len(A) != len(b) or     
        np.any(A[2:,0] != 0) or
        np.any(A[:-2,-1] != 0)):
            print("Incorrect tridiagonal structure.")
            return
    
    Arows = len(A)
    A = A.astype(float)
    b = b.astype(float)

    # making A into lower triangular matrix
    for i in range(Arows-1, 0, -1):
        b[i-1] = b[i-1] - b[i]*A[i-1][i]/A[i][i]
        A[i-1] = A[i-1] - A[i]*A[i-1][i]/A[i][i]

    # gauss elimination solving
    # note, np.linalg.solve() also will work
    x = np.zeros(Arows)
    k = Arows-1
    x[k] = b[k]/A[k][k]
    while k >= 0:
        x[k] = (b[k] - np.dot(A[k,k+1:],x[k+1:]))/A[k,k]
        k = k-1
    return x
    
def generalizedSolver(A, b):
    # check if input matrix is of correct form
    if (len(A) != len(b) or     
        np.any(A[2:,0] != 0) or
        np.any(A[:-2,-1] != 0)):
            print("Incorrect tridiagonal structure.")
            return
            
    A = A.astype(float)
    b = b.astype(float)
    
    # first getting rid of special delta value and epsilon value
    b[0] = b[0] - b[1]*A[0][2]/A[1][2]
    b[-1] = b[-1] - b[-2]*A[-1][-3]/A[-2][-3]
    
    A[0] = A[0] - A[1]*A[0][2]/A[1][2]
    A[-1] = A[-1] - A[-2]*A[-1][-3]/A[-2][-3]

    # then run same tridiagonal solver on new matrix
    x = tridiagonalSolver(A, b)
    return x

def betterSolver(A, b):
    # check if input matrix is of correct form
    if (len(A) != len(b) or     
        np.any(A[2:,0] != 0) or
        np.any(A[:-2,-1] != 0)):
            print("Incorrect tridiagonal structure.")
            return
    
    # perform L (lower triangular) and U (upper triangular) decomposition
    P,L,U = linalg.lu(A)
    
    # solve
    x = linalg.solve(L, b)
    v = linalg.solve(U, x)
    return v

anotherA = np.array([[-2, 1, 0, 0],
                    [1, -2, 1, 0],
                    [0, 1, -2, 1],
                    [0, 0, 1, -2]])
anotherB = np.array([[0.04],
                    [0.04],
                    [0.04],
                    [0.04]])

anotherA2 = np.array([[-2, 1, 5, 0],
                    [1, -2, 1, 0],
                    [0, 1, -2, 1],
                    [0, 7, 1, -2]])
anotherB2 = np.array([[0.04],
                    [0.04],
                    [0.04],
                    [0.04]])

tridiagonalSolver(anotherA, anotherB)
generalizedSolver(anotherA2, anotherB2)