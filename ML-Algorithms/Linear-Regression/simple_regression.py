#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 17:41:13 2020

@author: Explore Science and Mathematics
"""


import matplotlib.pyplot as plt
import numpy as np


# function to calculate the Mean Squared Error (Cost function)
def MSE(X, Y, m, b):
    N = len(X)
    
    mse = 0.0
   
    for (xi, yi) in zip(X, Y):
        fi = m*xi + b
        mse += (yi - fi)**2 
    return mse / N


# get the predictions
def predict(x, m, b):
    return m*x + b


# Gradient descent algoriythm
def update_weights(X, Y, m, b, lr):
    N = len(X)
    
    dm = 0.0
    db = 0.0
    
    # Calculate the gradients/ partial derivatives
    for (xi, yi) in zip(X, Y):
        dm +=  -2*xi*(yi - (m*xi + b))
        db +=  -2*(yi - (m*xi + b))
    
    # Update weights and bias
    m -= lr * (dm/N)
    b -= lr * (db/N)
    
    return m, b

# Find the weights and bias coefficients
def train(X, Y, lr = 1e-7, max_error = 20, num_iter = 100):
    cost_history = []
    
    # Initialize weights and bias to zero (Not necessary to be zero)
    m = 0.0
    b = 0.0
    
    interval = int(num_iter*0.01)
    print('[INFO] Training Started ...')
    
    for i in range(num_iter):
        # Run gradient descent to update the weights
        m, b = update_weights(X, Y, m, b, lr)
        
        # Calculate the error for the updated weights and bias
        error = MSE(X, Y, m, b)
        
        cost_history.append(error)
        
        # Print intermediate results
        if i % interval == 0:
            print('[DEBUG] Iteration: {} \t weights: {} \t bias: {} \t error: {}'.format(i, m, b, error))
        
        if error < max_error:
            print('[DEBUG] Iteration: {} \t weights: {} \t bias: {} \t error: {}'.format(i, m, b, error))
            print('[INFO] Early stopping ...')
            return m, b, cost_history
    print('[INFO] Training Completed ...')  
    return m, b, cost_history


# Plot the training results
def plot_results(history, noise_std):
    plot1 = plt.figure(1)
    plt.plot(history)
    plt.xlabel('number of iterations')
    plt.ylabel('Error - MSE')
    plt.title('Error variation')
    plt.ylim(0, noise_std**2*5)
    plt.show()
    
    
# Evaluvate how well model has learnt
def evaluate(X, Y, W , b):
    y_pred = predict(X, W, b)
    
    sst = np.sum((Y-Y.mean())**2)
    ssr = np.sum((y_pred-Y)**2)
    r2 = 1-(ssr/sst)
    return(r2)


# Main function
def main():
    lr = 0.0003 # learning rate
    epochs = 50000 # number of epochs/iterations should be trained
    noise_std = 50 # standard-deviation of the added noise
    
    X =np.random.sample(150)*50
   
    Y = 50 * X - 100
    
    noise = np.random.normal(0, noise_std, Y.shape)
    Y_noisy = Y + noise
    
    X_train, y_train, X_test, y_test = X[:100], Y_noisy[:100], X[100:], Y_noisy[100:]
    
    weight, bias, history = train(X_train, y_train, num_iter = epochs, lr=lr)
    
    print('r2 score: {}'.format(evaluate(X_test, y_test, weight, bias)))
    
    y_pred = np.array([predict(x, weight, bias) for x in X_test])
    
    plot2 = plt.figure(2)
    plt.scatter(X_test, y_test)
    plt.plot(X_test, y_pred, 'r')
    
    plot_results(history, noise_std)
    

if __name__ == "__main__": 
    main() 
    




        
        
    