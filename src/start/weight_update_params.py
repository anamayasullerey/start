"""
Weight update parameters.
The weight update parameter objects contain the parameters that are
required by the weight update algorithm.
"""
import numpy as np

class LearningRate(object):
    def __init__(self, alpha = 0.1, decay = "none", k = "0.001"):
        self.alpha0 = alpha
        self.alpha = alpha
        self.decay = decay
        self.k = k
        self.t = 0

    def update(self):
        self.t += 1
        if (self.decay == "step"):
            self.alpha -= self.k
        elif (self.decay == "exp"):
            self.alpha = self.alpha * np.exp(-self.k)
        elif (self.decay == "inverse"):  
            self.alpha = self.alpha0/(1+self.k*self.t)

class GradientDescentParams(object):
    weight_update_func_name = "gradient_descent"
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

class MomentumParams(object):
    weight_update_func_name = "momentum"
    def __init__(self, learning_rate, beta = 0.9):
        self.beta = beta
        self.learning_rate = learning_rate

class AdamParams(object):
    weight_update_func_name = "adam"
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 1
