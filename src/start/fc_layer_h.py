"""
fully connected layer
"""

import numpy as np
from . import layer

class FcLayer(layer.Layer):

    layer_name = "fc"
    layer_type = "hidden"
    
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
        self.parameters = ["weights", "bias"]

    def forward_calc(self, x):
        self.activations = np.dot(self.weights, x) + self.bias

    def backward_calc(self):
        #calculate dWeights
        batch_size = self.activations.shape[1]
        self.dweights = np.dot(self.dactivations, np.transpose(self.prev_layer.activations))/batch_size
        if (self.l2_loss_coeff != 0):
            self.dweights = self.dweights + ((self.l2_loss_coeff*self.weights)/batch_size)
        #calculate bias
        self.dbias = np.sum(self.dactivations, axis=1, keepdims=True)/batch_size

    def backward_grad(self):
        #calculate dactivation for previous layer
        self.prev_layer.dactivations = np.dot(np.transpose(self.weights), self.dactivations)

    def initialize_parameters(self):
        self.weights = np.random.randn(self.num_neurons, self.prev_layer.num_neurons)  * 0.01
        self.bias = np.zeros((self.num_neurons,1))
        self.dweights = np.zeros(self.weights.shape)
        self.dbias = np.zeros(self.bias.shape)

    def get_l2_loss(self):
        l2_loss = self.l2_loss_coeff*np.sum(np.square(self.weights))/2
        return l2_loss

    def print_forward(self):
        super().print_forward()
        print("weights:")
        print(self.weights)
        print("bias:")
        print(self.bias)

    def print_backward(self):
        super().print_backward()
        print("dweights:")
        print(self.dweights)
        print("dbias:")
        print(self.dbias)
        print("weights:")
        print(self.weights)
        print("bias:")
        print(self.bias)
