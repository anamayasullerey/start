"""
relu hidden layer
"""

import numpy as np
from . import activation_functions as af
from . import layer

class ReluLayer(layer.Layer):

    layer_name = "relu"
    layer_type = "hidden"

    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
    
    def check(self):
        self.check_num_neurons()

    def forward_calc(self, x):
        self.activations = af.relu(x)

    def backward_grad(self):
        self.prev_layer.dactivations = self.dactivations * af.relu_prime(self.prev_layer.activations)
