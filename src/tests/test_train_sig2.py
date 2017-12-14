import unittest
import nn_grad_test as nt
import numpy as np
import start.neural_network as nn
import start.layer_dict as ld
import start.weight_update_params as wup

class TestTrainSig2(unittest.TestCase):
        
    def test(self):
        net = nn.NeuralNetwork("test_net", 1)

        layer = ld.hdict["fc"](10)
        net.add_layer(layer)

        layer = ld.hdict["fc"](1)
        net.add_layer(layer)

        layer = ld.hdict["sigmoid"](1)
        net.add_layer(layer)

        layer = ld.odict["loss"]("sigmoid_cross_entropy_loss")
        net.add_layer(layer)

        np.random.seed(1)

        net.set_l2_loss_coeff(.001)        

        params = wup.GradientDescentParams(.2)
        net.set_weight_update_function(params)
        net.initialize_parameters()

        a = 0.8;

        for i in range(10000):
            x = (np.random.rand(1,32)*0.1  + 0.75)
            y = x > a
            net.train(x,y)
        
        x = 0.79 
        # print(net.predict(x))
        self.assertTrue(net.predict(x) < 0.5)
        x = 0.81
        # print(net.predict(x))
        self.assertTrue(net.predict(x) > 0.5)

