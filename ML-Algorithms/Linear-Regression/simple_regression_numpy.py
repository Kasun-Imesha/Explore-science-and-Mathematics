#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 14:12:05 2020

@author: Explore Science and Mathematics
"""


import matplotlib.pyplot as plt
import numpy as np


# function to calculate the Mean Squared Error (Cost function)
def MSE(X, Y, m, b):    
    square_sum = np.square(Y - (m*X + b))
   
    mse = np.mean(square_sum)
    return mse


# get the predictions
def predict(x, m, b):
    return m*x + b


# Gradient descent algoriythm
def update_weights(X, Y, m, b, lr):
    
    # Calculate the gradients/ partial derivatives
    dm = np.sum(-2*X*(Y - (m*X + b)))
    db = np.sum(-2*(Y - (m*X + b)))
     
    # Update weights and bias
    m -= lr * dm
    b -= lr * db
    
    return m, b


def train(X, Y, lr = 1e-7, max_error = 21, num_iter = 100):
    cost_history = []
    
    # Initialize weights and bias to zero (Not necessary to be zero)
    m = 0.0
    b = 0.0
    
    for i in range(num_iter):
        # Run gradient descent to update the weights
        m, b = update_weights(X, Y, m, b, lr)
        
        # Calculate the error for the updated weights and bias
        error = MSE(X, Y, m, b)
        
        cost_history.append(error)
        
        # Print results in every 10 iterations
        if i % 10 == 0:
            print('[DEBUG] Iteration: {} \t weights: {} \t bias: {} \t error: {}'.format(i, m, b, error))
        
        if error < max_error:
            print('[DEBUG] Iteration: {} \t weights: {} \t bias: {} \t error: {}'.format(i, m, b, error))
            print('[INFO] Early stopping ...')
            return m, b, cost_history
        
    return m, b, cost_history


# Plot the training results
def plot_results(history):
    plot1 = plt.figure(1)
    plt.plot(history)
    plt.xlabel('number of iterations')
    plt.ylabel('Error - MSE')
    plt.xlim(5, len(history))
    plt.ylim(0, 1000)
    plt.show()
    

# Main function
def main():
    X =np.random.sample(50)*50
   
    Y = 5 * X + 2
    
    noise = np.random.normal(0, 4.5, Y.shape)
    Y_noisy = Y + noise
    
    
    
    weight, bias, history = train(X, Y_noisy, num_iter = 1000, lr=0.000001)
    
    y_pred = np.array([predict(x, weight, bias) for x in X])
    
    plot2 = plt.figure(2)
    plt.scatter(X, Y_noisy)
    plt.plot(X, y_pred, 'r')
    
    plot_results(history)
    

if __name__ == "__main__": 
    main() 
    




        
        
    