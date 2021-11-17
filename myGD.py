#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 19:51:20 2021

@author: Dingran Yuan
"""

import numpy as np

def Gradient_Descent(gradient, A, b, init, learn_rate, n_iter, tolerance = 1e-06, dtype = "float64"): 
    """
    * gradient: the gradient function
    * A: observation input
    * b: output
    * learn_rate: the step length
    * n_iter: max iteration round
    * tolerance: convergence criteria
    """
    # Checking is the gradient is callable
    if not callable(gradient):
        raise TypeError("'gradient' must be a callable function.")
        
    # Initialising the data type for numpy arrays
    dtype_ = np.dtype(dtype)
    
    # Converting A and b into numpy arrays with data type
    A, b = np.array(A, dtype = dtype_), np.array(b, dtype=dtype_)
    if A.shape[0] != b.shape[0]:
        raise ValueError("'A' and 'b' lengths do not match.")
        
    # Initialising the value of the parameters
    vector = np.array(init, dtype = dtype_)
    
    # Initialising and checking the learning rate
    learn_rate = np.array(learn_rate, dtype = dtype_)
    if np.any(learn_rate <= 0):
        raise ValueError("'learn rate' must be greater than 0.")
        
    # Initialising and checking the max value of max iterations
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError("'n_iter' must be greater than 0.")
        
    # Initialising and checking the tolerance
    tolerance = np.array(tolerance, dtype = dtype_)
    if np.any(tolerance <= 0):
        raise ValueError("'tolerance' must be greater than 0.")
        
    # Gradient descent loop
    for _ in range(n_iter):
        print("iteration {}".format(_))
        # Calculating the descent value
        diff = learn_rate * np.array(gradient(A, b, vector), dtype = dtype_)
        #print(diff.shape)
        
        # Checking the convergence
        if np.all(np.abs(diff) <= tolerance):
            break
        
        # Update the parameter estimation
        vector -= diff
        
    return vector if vector.shape else vector.item()

def LS_Gradient(A, b, vector):
    """
    * A: input least square observation matrix
    * b: observation value
    * vector: parameter value
    """
    #print(vector.shape, b.shape, A.shape)
    gd = np.matmul(np.matmul(np.transpose(A), A), vector) - np.matmul(np.transpose(A), b)
    #gd=np.matmul(A.transpose(),np.matmul(A,vector)-b)
    #print(gd.shape)
    return gd

# Generate true answer 
mu, sigma = 0, 1
n, m = 50, 200
x = np.random.normal(mu, sigma, n)

# Randomly generate A
A = np.random.normal(mu, sigma, size = (m, n))

# Calculate b
b = np.matmul(A, x)
sigma2 = n/40
b += np.random.normal(0, sigma2, m)

# initialise vector
x0 = np.zeros((n,))

r = Gradient_Descent(LS_Gradient,A, b, x0, 0.003, 500)
print(np.linalg.norm(x-r))