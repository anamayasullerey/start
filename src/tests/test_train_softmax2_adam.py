import unittest
import nn_grad_test as nt
import numpy as np
import start.neural_network as nn
import start.layer_dict as ld
import start.weight_update_params as wup

class TestTrainSoftmax2Adam(unittest.TestCase):
        
    def test(self):
        net = nn.NeuralNetwork("test_net", 1)

        layer = ld.hdict["fc"](30)
        net.add_layer(layer)

        layer = ld.hdict["fc"](10)
        net.add_layer(layer)

        layer = ld.odict["softmax"](10)
        net.add_layer(layer)

        np.random.seed(1)

        params = wup.AdamParams(.1)
        net.set_weight_update_function(params)
        net.initialize_parameters()

        net.set_l2_loss_coeff(.001)        

        for i in range(100000):
            x = np.random.rand(1,32)
            y = np.array(x < 0.1)
            for j in range(1, 10):
              low = j * 0.1
              high = (j+1)*0.1
              yj = (x>=low) * (x<high)
              y = np.append(y, yj, axis=0)
            y = np.array(y, dtype=float)
            net.train(x,y)
        
        for j in range(0, 10):
            xl = j*0.1 + .02;
            xh = j*0.1 + .08;
            self.assertTrue(net.predict_classify(xl) == j)
            self.assertTrue(net.predict_classify(xh) == j)
