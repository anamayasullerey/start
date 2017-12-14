import nn_grad_test as nt
import numpy as np
import start.neural_network as nn
import start.layer_dict as ld
import start.weight_update_params as wup

class test_grad_x1_fc1_ms_reg(nt.NnGradTest):
        
    def define_nn(self):
        self.net = nn.NeuralNetwork("test_net", 1)

        self.layer = ld.hdict["fc"](1)
        self.net.add_layer(self.layer)

        self.layer = ld.odict["loss"]("linear_mean_squared_loss")
        self.net.add_layer(self.layer)

        self.net.set_l2_loss_coeff(1)

        np.random.seed(1)

        self.params = wup.GradientDescentParams(0.1)
        self.net.set_weight_update_function(self.params)
        self.net.initialize_parameters()
        self.net.layers[1].weights[0,0] = 10
    
    def set_training_example(self):
        self.x = np.array([[2]])
        self.y = np.array([[10]])
    
    def extra_checks(self):
        if ((self.net.layers[1].dweights[0][0]) != 30):
            print("Error: Expected gradient = 30.0")
            print("     : Backprop gradient = " + str(self.net.layers[1].dweights[0][0]))
            return False
        else:
            return True



