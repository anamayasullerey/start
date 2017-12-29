import unittest
import nn_grad_test as nt
import numpy as np
import start.neural_network as nn
import start.layer_dict as ld
import start.weight_update_params as wup

class TestTrainLin2(unittest.TestCase):
        
    def test(self):
        net = nn.NeuralNetwork("test_net", 4)

        layer = ld.hdict["fc"](10)
        net.add_layer(layer)

        layer = ld.hdict["fc"](40)
        net.add_layer(layer)

        layer = ld.hdict["fc"](4)
        net.add_layer(layer)

        layer = ld.odict["loss"]("linear_mean_squared_loss")
        net.add_layer(layer)

        net.set_l2_loss_coeff(.001)        

        np.random.seed(1)

        learning_rate = wup.LearningRate(alpha=0.01)
        params = wup.GradientDescentParams(learning_rate)
        net.set_weight_update_function(params)
        net.initialize_parameters()

        a = np.array([[1], [2], [3], [4]])
        b = np.array([[4], [5], [6], [7]])

        for i in range(1000):
            x = (np.random.rand(4,32) - 0.5) * 10 
            y = a * x + b
            net.train(x,y)
        
        x = np.array([[10], [10], [10], [10]])
        y_exp = a * x + b
        self.assertTrue(((np.absolute(net.predict(x) - y_exp)/y_exp) < 0.1).all())


