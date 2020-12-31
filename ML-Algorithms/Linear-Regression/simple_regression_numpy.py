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
    dm = np.mean(-2*X*(Y - (m*X + b)))
    db = np.mean(-2*(Y - (m*X + b)))
     
    # Update weights and bias
    m -= lr * dm
    b -= lr * db
    
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
    




        
        
    