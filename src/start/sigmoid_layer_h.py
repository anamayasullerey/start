"""
sigmoid hidden layer
"""

from . import activation_functions as af
from . import layer

class SigmoidLayer(layer.Layer):

    layer_name = "sigmoid"
    layer_type = "hidden"

    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
    
    def check(self):
        self.check_num_neurons()

    def forward_calc(self, x):
        self.activations = af.sigmoid(x)

    def backward_grad(self):
        self.prev_layer.dactivations = self.dactivations * self.activations * (1 - self.activations)
