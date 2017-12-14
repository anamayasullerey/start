"""
This layer defines the loss function.
This layer has identity activation function.
"""

from . import layer
from . import loss_functions as lf

class LossLayer(layer.Layer):

    layer_name = "loss"
    layer_type = "output"
    
    def __init__(self, loss_function_name):
        super().__init__()
        self.loss_func = getattr(lf, loss_function_name)
        self.loss_func_prime = getattr(lf, loss_function_name + "_prime")

    def forward_calc(self, x):
        self.activations = x

    # backprop calculations
    def backward_grad(self):
        # Set the dactivation based on loss function
        self.prev_layer.dactivations = self.loss_func_prime(self.y, self.activations)
        
    def backward_prop(self, y):
        # copy y in the layer. It is later used during calculation
        # gradients 
        self.y = y
        super().backward_prop()

    # backprop calculations
    def loss(self, y):
        loss_value = self.loss_func(y, self.activations)
        return loss_value

    def print_backward(self):
        pass
