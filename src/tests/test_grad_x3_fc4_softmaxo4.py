import nn_grad_test as nt
import numpy as np
import start.neural_network as nn
import start.layer_dict as ld
import start.weight_update_params as wup

class test_grad_x3_fc1_sigm1_sigce(nt.NnGradTest):
        
    def define_nn(self):
        self.net = nn.NeuralNetwork("test_net", 3)

        self.layer = ld.hdict["fc"](4)
        self.net.add_layer(self.layer)

        self.layer = ld.odict["softmax"](4)
        self.net.add_layer(self.layer)

    def set_training_example(self):
        self.x = np.array([[1], [2], [3]])
        self.y = np.array([[1], [0], [0], [0]])
