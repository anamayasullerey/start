import nn_grad_test as nt
import numpy as np
import start.neural_network as nn
import start.layer_dict as ld
import start.weight_update_params as wup

class test_grad_x3_fc1_sigm1_sigce(nt.NnGradTest):
        
    def define_nn(self):
        self.net = nn.NeuralNetwork("test_net", 1)

        self.layer = ld.hdict["fc"](1)
        self.net.add_layer(self.layer)

        self.layer = ld.hdict["sigmoid"](1)
        self.net.add_layer(self.layer)

        self.layer = ld.odict["loss"]("sigmoid_cross_entropy_loss")
        self.net.add_layer(self.layer)

        np.random.seed(1)

        self.params = wup.GradientDescentParams(0.1)
        self.net.set_weight_update_function(self.params)
        self.net.initialize_parameters()
    
    def set_training_example(self):
        self.x = np.array([[0.5]])
        self.y = np.array([[0.5]])
