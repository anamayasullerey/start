from . import input_layer
from . import weight_update_functions as wuf
import numpy as np

class NeuralNetwork(object):

    # Input layer is automaticaly created when a NeuralNetwork is created
    def __init__(self, name, numInputs):
        self.name = name
        self.layers = []
        il = input_layer.InputLayer(numInputs)
        il.layer_num = 0
        self.layers.append(il)

    def add_layer(self, layer):
        layer.layer_num = len(self.layers)
        self.layers[-1].next_layer = layer
        layer.prev_layer = self.layers[-1]
        layer.next_layer = None
        self.layers.append(layer)

    def check_arch(self):
        for layer in self.layers:
            layer.check()          
        assert type(self.layers[0]).layer_type is "input", "Error: First layer (layer 0) is not an input layer"
        assert type(self.layers[-1]).layer_type is "output", "Error: Last layer is not an output layer"
        for i in range(1, len(self.layers)-2):
            error_msg = "Layer number " + str(i) + ": layer type :" + type(self.layers[i]).layer_type + "\n"
            error_msg += "Error: Intermediate layer must be of type hidden"
            assert type(self.layers[i]).layer_type is "hidden", error_msg
                   
        
    def set_weight_update_function(self, wu_params):
        self.weight_update_func= getattr(wuf, wu_params.weight_update_func_name)
        self.weight_update_init_func = getattr(wuf, wu_params.weight_update_func_name + "_init")
        self.wu_params = wu_params

    def initialize_parameters(self):
        for i in range(len(self.layers)):
            self.layers[i].initialize_parameters()
            self.weight_update_init_func(self.layers[i])

    # x should be a 2D numpy array
    def forward_prop(self, x):
        self.layers[0].forward_prop(x)

    # y should be a 2D numpy array
    def backward_prop(self, y):
        self.layers[-1].backward_prop(y)

    def weight_update(self):
        for l in self.layers:
            self.weight_update_func(l, self.wu_params)

    def train(self, x, y):
        self.forward_prop(x)
        self.backward_prop(y)
        self.weight_update()

    def set_l2_loss_coeff(self, l2_loss_coeff):
        for l in self.layers:
            l.set_l2_loss_coeff(l2_loss_coeff)

    def loss(self, y):
        loss_value = self.layers[-1].loss(y)
        l2_regu_loss = 0
        for l in self.layers:
            l2_regu_loss += l.get_l2_loss()
        loss_value += l2_regu_loss
        return loss_value

    def predict(self, x):
        self.forward_prop(x)
        return self.layers[-1].get_activations()

    def predict_classify(self, x):
        self.forward_prop(x)
        return self.layers[-1].get_labels()

    def classification_accuracy(self, x, y_class):
        y_pred = self.predict_classify(x)
        total_correct = np.sum(y_pred == y_class)
        accuracy = total_correct * 100 / y_class.shape[0]
        return accuracy
   
    def print_state(self):
        print("Printing state")
        for i in range(len(self.layers)):
            self.layers[i].print_forward()
            self.layers[i].print_backward()

    def check_gradient(self, x, y):
        # numerically calculate gradients for each parameter
        # weights are not updated during the check
        # first calculate the model gradients by running forward and 
        # backward pass
        # Note: Currently the gradient checker works for 2D numpy arrays
        #       Parameters with higher dimentions will give errors.
        self.forward_prop(x)
        self.backward_prop(y)
        self.cost = self.loss(y)

        for l in self.layers:
            l.dparams_numerical = {}
            for param in l.parameters:
                # Neumerical derivatives are stored in a dictionary
                # Initialize the parameter in the dictionary
                l.dparams_numerical[param] = np.zeros(l.__dict__[param].shape)
                for i in range(l.dparams_numerical[param].shape[0]):
                    for j in range(l.dparams_numerical[param].shape[1]):
                        indices = (i,j)
                        numerical_grad = self.calculate_numerical_gradient(x, y, l, param, indices)
                        l.dparams_numerical[param][indices] = numerical_grad
                        if not NeuralNetwork.error_chk(l.__dict__["d" + param][indices], numerical_grad):
                            print ("Error: Gradient check failed for layer " + str(l.layer_num))
                            print ("       parameter = " + str(param))
                            print ("       index = " + str(indices))
                            print ("       gradient = " + str(l.__dict__["d" + param][indices]))
                            print ("       numerical gradient = " + str(numerical_grad))
                            return False
        return True                   

    def calculate_numerical_gradient(self, x, y, layer, param, indices, epsilon=1e-7):
        # store original value of the parameter
        orig_value = layer.__dict__[param][indices]
        
        # param + epsilon
        layer.__dict__[param][indices] += epsilon
        self.forward_prop(x)
        cost_plus = self.loss(y)
        
        # param - epsilon
        layer.__dict__[param][indices] -= 2*epsilon
        self.forward_prop(x)
        cost_minus = self.loss(y)
        
        # restore parameter
        layer.__dict__[param][indices] = orig_value
        
        # return numerical gradient
        return (cost_plus-cost_minus)/(2*epsilon)

    # This function checks if the calculated and neumerical gradients are
    # close enough. It may now work for very small values of gradients.
    @staticmethod
    def error_chk(value0, value1, margin_fraction=1e-4):
        numerator = np.linalg.norm(value0 - value1)
        denominator = np.linalg.norm(value0) + np.linalg.norm(value1)
        return (numerator <= margin_fraction * denominator)
