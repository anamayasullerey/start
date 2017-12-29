"""
Weight update  functions

A weight update functions take layer as an input and update that layer's 
parameters. These update functions updates all the parameters defined
in layer.parameters[] list. The caculation of the derivatives is done
during the backprop. Any weight update function uses the calculated 
derivatives to do the parameter update.

Any state required by the weight update function is stored in the layer
itself. Each weight update funtion has an associated initialization function
for this state information.

Weight update functions also have an assoctiated clsss that defines the
weight update parameters. An object of this class is passed along with
the layer for weigh update.  
"""
import numpy as np

def gradient_descent(layer, wu_params):
    for param in layer.parameters:
        layer.__dict__[param] -= wu_params.learning_rate.alpha * layer.__dict__["d" + param]

def gradient_descent_init(layer):
    pass

def momentum(layer, wu_params):
    for param in layer.parameters:
        layer.velocity[param] = wu_params.beta*layer.velocity[param] + (1-wu_params.beta)*layer.__dict__["d" + param]
        layer.__dict__[param] -= wu_params.learning_rate.alpha * layer.velocity[param]
    
def momentum_init(layer):
    layer.velocity = {}
    layer.sq_grad = {}
    for param in layer.parameters:
        layer.velocity[param] = np.zeros(layer.__dict__[param].shape)
        layer.sq_grad[param] = np.zeros(layer.__dict__[param].shape)

def adam(layer, wu_params):
    # The adjustments below are for completeness sake.
    adj1 = 1/(1 - np.power(wu_params.beta1, wu_params.t))
    adj2 = 1/(1 - np.power(wu_params.beta2, wu_params.t))
    wu_params.t += 1
    for param in layer.parameters:
        # calculate velocity
        layer.velocity[param] = wu_params.beta1*layer.velocity[param] + (1-wu_params.beta1)*layer.__dict__["d" + param]

        # calculate square of gradients
        layer.sq_grad[param] = wu_params.beta2*layer.sq_grad[param] + (1 - wu_params.beta2)*np.power(layer.__dict__["d" + param], 2)

        # adjustment
        weight_velocity_adj = layer.velocity[param] * adj1
        sq_grad_adj = layer.sq_grad[param] * adj2

        layer.__dict__[param] -= wu_params.learning_rate.alpha * weight_velocity_adj/(np.sqrt(sq_grad_adj) + wu_params.epsilon)
    
    
def adam_init(layer):
    layer.velocity = {}
    layer.sq_grad = {}
    for param in layer.parameters:
        layer.velocity[param] =  np.zeros(layer.__dict__[param].shape)
        layer.sq_grad[param] =  np.zeros(layer.__dict__[param].shape)
