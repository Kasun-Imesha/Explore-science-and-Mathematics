#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 11:24:18 2020

@author: Explore Science and Mathematics
"""

from sklearn.model_selection import train_test_split # to split the dataset into train-set and test-set
import matplotlib.pyplot as plt # to plot the graphs
from sklearn import datasets # to import 'boston housing' dataset
import numpy as np # for math operations
import pandas as pd # to import 'power plant' dataset


# z-score normalization
def z_normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean)/std
    return X

# Min-Max normalization
def minmax_normalize(X):
    Max = np.max(X, axis=0)
    Min = np.min(X, axis=0)
    X = (X - Min)/(Max - Min)
    return X    

# function to calculate the Mean Squared Error (Cost function)
def MSE(X, Y, W):
        
    square_sum = np.square(Y - np.dot(X, W)) 
    mse = np.mean(square_sum)
   
    return mse


# get the predictions
def predict(X, W):
    return np.dot(X,W)


# Gradient descent algoriythm
def update_weights(X, Y, W, lr):
    N = len(Y)
    
    # Calculate the gradients/ partial derivatives
    dW = (-2/N)*np.dot(X.T,(Y - np.dot(X,W)))
     
    # Update weights and bias
    W -= lr * dW
    
    return W


# Find the weights and bias coefficients
def train(X, Y, lr = 1e-7, max_error = 20, num_iter = 100):
    cost_history = []
    
    N = X.shape[1]
    
    # Initialize weights and bias to zero (Not necessary to be zero)
    W = np.zeros((N,1))
    
    interval = int(num_iter*0.01)
    
    print('[INFO] Training Started ...')
    for i in range(num_iter):
        # Run gradient descent to update the weights
        W = update_weights(X, Y, W, lr)
        
        # Calculate the error for the updated weights and bias
        error = MSE(X, Y, W)
        
        cost_history.append(error)
        
        # Print intermediate results
        if i % interval == 0:
            print('[DEBUG] Iteration: {} \t weights: {} \t error: {}'.format(i, W.ravel(), error))
        
        if error < max_error:
            print('[DEBUG] Iteration: {} \t weights: {} \t error: {}'.format(i, W.ravel(), error))
            print('[INFO] Early stopping ...')
            return W, cost_history
    print('[INFO] Training Completed ...')
    return W, cost_history

# Plot the training results
def plot_results(history):
    plot1 = plt.figure(1)
    plt.plot(history)
    plt.xlabel('number of iterations')
    plt.ylabel('Error - MSE')
    plt.title('Error variation')
    plt.ylim(0, 1000)
    plt.show()
    
# Evaluvate how well model has learnt
def evaluate(X,Y,W):
    y_pred = predict(X, W)
    
    sst = np.sum((Y-Y.mean())**2)
    ssr = np.sum((y_pred-Y)**2)
    r2 = 1-(ssr/sst)
    return(r2)
    
    

def main():
    lr = 0.001 # learning rate
    epochs = 10000 # number of epochs/iterations should be trained
    
    # load the boston dataset 
    boston = datasets.load_boston(return_X_y=False) 
      
    # defining feature matrix(X) and response vector(y) 
    X = boston.data 
    y = boston.target 
    
    y = np.expand_dims(y, axis=1)
    X = z_normalize(X)
    
    # add extra column of 1's to merge the bias
    ones = np.ones((len(X), 1))
    X = np.hstack((X, ones))
      
    # splitting X and y into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                        random_state=1) 
    
    print('Part of the dataset:')
    print(X_train[:,:5])
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    weights, history = train(X_train, y_train, num_iter = epochs, lr=lr)
    
    print('r2 score: {}'.format(evaluate(X_test,y_test,weights)))
    
    plot_results(history)
    

def main2():
    lr = 0.001 # learning rate
    epochs = 10000 # number of epochs/iterations should be trained
    
    # load the power-plant dataet
    data = pd.read_excel('Data/Folds5x2_pp.xlsx')
    
    print('Part of the dataset:')
    print(data.head(5))
    
    data = data.to_numpy()
    # print(data.shape)
    
    X = data[:,:4]
    y = data[:,-1]
    
    y = np.expand_dims(y, axis=1)
    
    # Data normalization
    X = z_normalize(X)
    
    # add extra column of 1's to merge the bias
    ones = np.ones((len(X), 1))
    X = np.hstack((X, ones))
    
    # splitting X and y into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                        random_state=1)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    weights, history = train(X_train, y_train, num_iter = epochs, lr=lr)
    
    print('r2 score: {}'.format(evaluate(X_test,y_test,weights)))
    
    plot_results(history)
    
    
if __name__ == "__main__": 
    print('----'*30)
    print("[INSTRUCTION] For 'Boston-housing' dataset demo uncomment main() and for 'power-plant' dataset demo uncomment main2()")
    print('----'*30)
    
    # main() 
    main2()




        
        
    