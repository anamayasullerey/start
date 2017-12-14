"""
Input layer of a neural network.
"""

from . import layer

class InputLayer(layer.Layer):

    layer_name = "input"
    layer_type = "input"
    
    def __init__(self, num_inputs):
        super().__init__()
        self.num_neurons = num_inputs

    def forward_calc(self, x):
        self.activations = x

    def backward_calc(self):
        print("Error: Network not stitched propertly. Backprop should not reach input layer.")

    def print_backward(self):
        pass
