"""
Loss functions and their derivatives

- Loss functions calculate overall loss
- Loss derivatives calculate loss per output neuron
  as required by the backpropagation

  # y is the observed output for the training example
  # a is the predicted output

"""
import numpy as np

def clip(x):
    epsilon = 1e-11
    return np.clip(x, epsilon, 1 - epsilon)

def sigmoid_cross_entropy_loss(y, a):
    a_ = clip(a)
    return  - np.sum((y*np.log(a_) + (1-y)*np.log(1-a_)))/y.shape[1]

def sigmoid_cross_entropy_loss_prime(y, a):
    a_ = clip(a)
    return - (np.divide(y, a_) - np.divide(1-y, 1-a_))  

def softmax_cross_entropy_loss(y, a):
    a_ = clip(a)
    return - np.sum(y*np.log(a_))/y.shape[1]

def softmax_cross_entropy_loss_prime(y, a):
    a_ = clip(a)
    return -np.divide(y, a_)

def linear_mean_squared_loss(y, a):
    return np.sum(np.power((a-y), 2))/(2*y.shape[1])

def linear_mean_squared_loss_prime(y, a):
    return (a-y)