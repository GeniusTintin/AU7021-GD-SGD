#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 19:50:57 2021

@author: Dingran Yuan
"""

import numpy as np
import matplotlib.pyplot as plt

def SGD(gradient, A, b, real_answer, init, learn_rate, n_iter, batch_size = 1, random_state = None, tolerance = 1e-06, dtype = "float64"):
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
    
    # Concatenating Ab
    Ab = np.c_[A.reshape(n_training, -1), b.reshape(n_training, 1)]
        
    # Initialising the value of the parameters
    x_hat = np.array(init, dtype = dtype_)
    
    # Initialising random number generator
    seed = None if random_state is None else int(random_state)
    random_number_generator = np.random.default_rng(seed = seed)
    
    # Initialising the learning rate
    learn_rate = np.array(learn_rate, dtype = dtype_)
    if np.any(learn_rate <= 0):
        raise ValueError("'learn_rate' must be greater than 0.")
        
    # Inisitlising the batch size
    batch_size = int(batch_size)
    if not (batch_size > 0 and batch_size <= n_training):
        raise ValueError("'batch_size' must be greater than 0, and less than or equal to the number of training samples.")
        
    # Initialising the n_iter
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError("'n_iter' must be greater than 0.")
        
    # Initialising the tolerance for convergence
    tolerance = np.array(tolerance, dtype = dtype_)
    if np.any(tolerance <= 0):
        raise ValueError("'tolerance' must be greater than 0.")
        
    # Initialising breaking condition
    n_accept = 0
    
    # Visualisation data
    error_array = []
    diff_array = []
    objective_array = []
        
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

            grad = np.array(gradient(A_batch, b_batch, x_hat), dtype = dtype_)
            diff = learn_rate * grad
            
            # Falling in the acceptance range 10 time to stop
            if np.all(np.abs(diff) <= tolerance):
                n_accept += 1
                break
            else:
                n_accept = 0
            
            x_hat -= diff
            error_array.append(np.linalg.norm(real_answer - x_hat))
            diff_array.append(np.linalg.norm(diff))
            objective_array.append(np.linalg.norm(np.matmul(A,x_hat) - b))
    
    return x_hat, diff_array, error_array, objective_array 
    #if x_hat.shape else x_hat.item(), diff_array, error_array, objective_array

# The gradient function of Least Square problem
def LS_Gradient(A, b, vector):
    """
    * A: input least square observation matrix
    * b: observation value
    * vector: parameter value
    """
    gd = np.matmul(np.matmul(np.transpose(A), A), vector) - np.matmul(np.transpose(A), b)
    return gd

# Function of Pseudo Inverse
def Pseudo_Inverse(A, b):
    x_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.transpose(),A)), A.transpose()), b)
    return x_hat

def Pseudo_Inverse2(A,b):
    xhat = np.matmul(A.transpose(),np.matmul(np.linalg.inv(np.matmul(A,A.transpose())),b))
    return xhat

# Generate true answer 
mu, sigma = 0, 1
#n_seq, m = [50, 300, 500, 1000, 2000], 200
n_seq, m = [300,500,1000,2000], 200

# Solving the problem with different n
for n in n_seq:
    x_bar = np.random.normal(mu, sigma, n)

    # Randomly generate A
    A = np.random.normal(mu, sigma, size = (m, n))

    # Calculate b
    b = np.matmul(A, x_bar)
    sigma2 = n/40
    b += np.random.normal(0, sigma2, m)

    # initialise vector
    #x0 = 5 * np.ones((n,))
    # Random init
    x0 = np.random.normal(mu, sigma, size = (n,))
    #print(x0)
    
    learn_rate = 0.0005
    batch_size = 20

    # Use Stochastic Gradient Descent to calculate x_hat
    x_hat, diff_array, error_array, objective_array = SGD(LS_Gradient, A, b, x_bar, x0, learn_rate, 1e4, batch_size=batch_size, tolerance=1e-06 )
    
    r = Pseudo_Inverse2(A, b)
    print("||x_hat-r|| = {}".format(np.linalg.norm(r-x_hat)))
    # Visualisation process
    fig = plt.figure()
    axs1 = fig.add_subplot(2,3,1)
    axs2 = fig.add_subplot(2,3,2)
    axs3 = fig.add_subplot(2,3,3)
    axs4 = fig.add_subplot(2,3,(4,6))
    axs1.plot(np.array(error_array), 'go-', markersize = 3)
    axs2.plot(np.array(diff_array), 'o-', markersize = 3)
    axs3.plot(np.array(objective_array),'yo-', markersize = 3)
    axs4.plot(x_bar,'go-',markersize = 3, label='ground-truth')
    axs4.plot(r,'co-',markersize = 5, label='pseudo inverse')
    axs4.plot(x_hat,'ro-',markersize = 3, label = 'SGD')
    axs4.legend()
    axs1.grid(); axs2.grid(); axs3.grid(); axs4.grid();
    axs1.set_xlabel("Iterations"); axs2.set_xlabel("Iterations"); axs3.set_xlabel("Iterations"); axs4.set_xlabel("Samples")
    axs1.set_ylabel("Cost error"); axs2.set_ylabel("Step length"); axs3.set_ylabel("Objective value"); axs4.set_ylabel("Parameter values")
    axs1.set_title("$\||x^k-\\bar{x}\||$"); axs2.set_title("Step length over iterations"); axs3.set_title("Objective value over iterations"); 
    fig.suptitle("learning rate = {}, n = {}, batch size = {}".format(learn_rate, A.shape[1], batch_size))
    fig.set_size_inches(15, 8)
    plt.show()
    
    
    print("||x_hat-x_bar|| = {}".format(np.linalg.norm(x_bar-x_hat)))
    
# Result of Pseudo Inverse
#r = Pseudo_Inverse2(A, b)
#print("||x_hat - r|| = {}".format(np.linalg.norm(x_hat-r)))