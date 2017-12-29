import nn_grad_test as nt
import numpy as np
import start.neural_network as nn
import start.layer_dict as ld
import start.weight_update_params as wup

class test_x3_fc1_sigmo1(nt.NnGradTest):
        
    def define_nn(self):
        self.net = nn.NeuralNetwork("test_net", 1)

        self.layer = ld.hdict["fc"](1)
        self.net.add_layer(self.layer)

        self.layer = ld.odict["sigmoid"](1)
        self.net.add_layer(self.layer)

    def set_training_example(self):
        self.x = np.array([[0.5]])
        self.y = np.array([[0.5]])
