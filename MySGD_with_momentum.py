#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:41:00 2021

@author: tintin
"""

import numpy as np
import matplotlib.pyplot as plt

def SGD(gradient, A, b, real_answer, init, learn_rate, n_iter, decay_rate = 0.0, batch_size = 1, random_state = None, tolerance = 1e-06, dtype = "float64"):
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
    n_training = A.shape[0]
    if n_training != b.shape[0]:
        raise ValueError("'A' and 'b' lengths do not match.")
    
    # concadinate Ab
    Ab = np.c_[A.reshape(n_training, -1), b.reshape(n_training, 1)]
        
    # Initialising the value of the parameters
    x_hat = np.array(init, dtype = dtype_)
    
    # Initialising random number generator
    seed = None if random_state is None else int(random_state)
    random_number_generator = np.random.default_rng(seed = seed)
    
    learn_rate = np.array(learn_rate, dtype = dtype_)
    if np.any(learn_rate <= 0):
        raise ValueError("'learn_rate' must be greater than 0.")
        
    batch_size = int(batch_size)
    if not (batch_size > 0 and batch_size <= n_training):
        raise ValueError("'batch_size' must be greater than 0, and less than or equal to the number of training samples.")
        
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError("'n_iter' must be greater than 0.")
        
    tolerance = np.array(tolerance, dtype = dtype_)
    if np.any(tolerance <= 0):
        raise ValueError("'tolerance' must be greater than 0.")
        
    # Initialise breaking condition
    n_accept = 0
    
    # Visualisation
    error_array = []
    
    # Setting the difference to zero the fist iteration
    diff = 0
        
    # Performance Stochastic Gradient Descent loop
    for _ in range(n_iter):
        print("iteration {}".format(_))
        if n_accept > 10:
            break
        # Shuffle Ab
        random_number_generator.shuffle(Ab)
        
        # Minibatch move
        for start in range(0, n_training, batch_size):
            stop = start + batch_size
            A_batch, b_batch = Ab[start:stop, :-1], Ab[start:stop, -1:]
            b_batch = b_batch.ravel()
            #print(A_batch.shape, b_batch.shape)
            grad = np.array(gradient(A_batch, b_batch, x_hat), dtype = dtype_)
            diff = learn_rate * grad - decay_rate * diff
            #print(diff.shape, grad.shape)
            
            if np.all(np.abs(diff) <= tolerance):
                n_accept += 1
                break
            else:
                n_accept = 0
                print(np.max(diff))
            
            x_hat -= diff
            error_array.append(np.linalg.norm(real_answer - x_hat))
    
    plt.plot(error_array)
    return x_hat if x_hat.shape else x_hat.item()


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

def Pseudo_Inverse(A,b):
    x_hat = np.matmul(np.matmul(A.transpose(),np.linalg.inv(np.matmul(A,A.transpose()))),b)
    # x_hat = 
    return x_hat

# Generate true answer 
mu, sigma = 0, 1
n_seq, m = [50, 300, 500, 1000, 2000], 200
n = n_seq[4]
x_bar = np.random.normal(mu, sigma, n)

# Randomly generate A
A = np.random.normal(mu, sigma, size = (m, n))

# Calculate b
b = np.matmul(A, x_bar)
sigma2 = n/40
b += np.random.normal(0, sigma2, m)

# initialise vector
x0 = np.zeros((n,))
# Random init
#x0 = np.random.normal(mu, sigma, size = (n,))
#print(x0)

r = SGD(LS_Gradient, A, b, x_bar, x0, 0.0003, 1e4, decay_rate = 0.8, batch_size=20, tolerance=1e-06 )
r0 = Pseudo_Inverse(A, b)
print(np.linalg.norm(x_bar-r))
print(np.linalg.norm(r-r0))
#plt.plot(error_array)