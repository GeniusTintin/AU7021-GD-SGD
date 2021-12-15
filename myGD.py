#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 19:51:20 2021

@author: Dingran Yuan
"""

import numpy as np
import matplotlib.pyplot as plt

def Gradient_Descent(gradient, A, b, real_value, init, learn_rate, n_iter, tolerance = 1e-06, dtype = "float64"): 
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
    x_hat = np.array(init, dtype = dtype_)
    
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
        
    # Initialise breaking condition
    n_accept = 0
        
    # Visualisation data
    error_array = []
    diff_array = []
    objective_array = []
        
    # Gradient Descent loop
    for _ in range(n_iter):
        print("iteration {}".format(_))
        # Calculating the descent value
        diff = learn_rate * np.array(gradient(A, b, x_hat), dtype = dtype_)
        #print(diff.shape)
        
        # Checking the convergence
        if np.all(np.abs(diff) <= tolerance):
            n_accept += 1
        else:
            n_accept = 0
            
        if n_accept >= 10:
            break
        
        # Update the parameter estimation
        x_hat -= diff
        error_array.append(np.linalg.norm(real_value -x_hat))
        diff_array.append(np.linalg.norm(diff))
        objective_array.append(np.linalg.norm(np.matmul(A,x_hat) - b))
        
    # Visualisation process
    fig, (axs1, axs2,axs3) = plt.subplots(1, 3)
    axs1.plot(np.array(error_array), 'go-', markersize = 3)
    axs2.plot(np.array(diff_array), 'o-', markersize = 3)
    axs3.plot(np.array(objective_array),'yo-', markersize = 3)

    axs1.grid(); axs2.grid(); axs3.grid();
    axs1.set_xlabel("Iterations"); axs2.set_xlabel("Iterations"); axs3.set_xlabel("Iterations")
    axs1.set_ylabel("Cost error"); axs2.set_ylabel("Step length"); axs3.set_ylabel("Objective value")
    axs1.set_title("$\||x^k-\\bar{x}\||$"); axs2.set_title("Step length over iterations"); axs3.set_title("Objective value over iterations")
    fig.suptitle("learning rate = {}".format(learn_rate))
    fig.set_size_inches(15, 4.5)
    plt.show()
    # fig.savefig('test.png', dpi=100)
    
    return x_hat if x_hat.shape else x_hat.item()

# The gradient function of Least Square problem
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

# Function of Pseudo Inverse
def Pseudo_Inverse(A, b):
    x_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.transpose(),A)), A.transpose()), b)
    return x_hat

# Generate true answer 
mu, sigma = 0, 1
n, m = 50, 200
x_bar = np.random.normal(mu, sigma, n)

# Randomly generate A
A = np.random.normal(mu, sigma, size = (m, n))

# Calculate b
b = np.matmul(A, x_bar)
sigma2 = n/40
b += np.random.normal(0, sigma2, m)

# initialise vector
x0 = np.zeros((n,))

# Use Gradient Descent to calculate x_hat
x_hat = Gradient_Descent(LS_Gradient,A, b, x_bar, x0, 0.001, 1e4)
# Result of Pseudo Inverse
r = Pseudo_Inverse(A, b)

# Print the estimation root square error and the root square error w.r.t. pseudo inverse
print(np.linalg.norm(x_bar-x_hat))
print(np.linalg.norm(x_hat - r))