import unittest
import nn_grad_test as nt
import numpy as np
import start.neural_network as nn
import start.layer_dict as ld
import start.weight_update_params as wup

class TestTrainSoftmax1Adam(unittest.TestCase):
        
    def test(self):
        net = nn.NeuralNetwork("test_net", 1)

        layer = ld.hdict["fc"](2)
        net.add_layer(layer)

        layer = ld.odict["softmax"](2)
        net.add_layer(layer)

        np.random.seed(1)

        params = wup.AdamParams(.1)
        net.set_weight_update_function(params)
        net.initialize_parameters()

        net.set_l2_loss_coeff(.001)        

        for i in range(10000):
            x = (np.random.rand(1,32)*0.1 + 0.75)
            y = x < 0.8
            y = np.append(y, x>=0.8, axis=0)
            y = np.array(y, dtype=float)
            net.train(x,y)
        
        x = 0.79
        self.assertTrue(net.predict_classify(x) == 0)
        x = 0.81
        self.assertTrue(net.predict_classify(x) == 1)
