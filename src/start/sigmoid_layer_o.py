"""
sigmoid output layer
"""

import numpy as np
from . import loss_functions as lf
from . import activation_functions as af
from . import layer

class SigmoidLayer(layer.Layer):

    layer_name = "sigmoid"
    layer_type = "output"
    
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
    
    def check(self):
        self.check_num_neurons()

    def forward_calc(self, x):
        self.activations = af.sigmoid(x)

    def backward_grad(self):
        self.prev_layer.dactivations = (self.activations - self.y)

    def backward_prop(self, y):
        self.y = y
        super().backward_prop()

    # backprop calculations
    def loss(self, y):
        loss_value = lf.sigmoid_cross_entropy_loss(y, self.activations)
        return loss_value

    def print_backward(self):
        pass
