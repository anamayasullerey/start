import unittest
import start.weight_update_params as wup
import numpy as np

class NnGradTest(unittest.TestCase):
        
    def define_nn(self):
        pass
    
    def set_training_example(self):
        pass
    
    def initialize(self):
        pass

    def test(self):
        self.define_nn();

        self.learning_rate = wup.LearningRate(alpha=0.1)
        self.params = wup.GradientDescentParams(self.learning_rate)
        self.net.set_weight_update_function(self.params)
        np.random.seed(1)
        self.net.initialize_parameters()
        self.initialize()

        self.set_training_example();
        self.assertTrue(self.net.check_gradient(self.x, self.y));
        self.assertTrue(self.extra_checks())

    def check_gradient(self):
        return self.net.check_gradient(self.x, self.y);
        
    def extra_checks(self):
        return True
    
if __name__ == '__main__':
    unittest.main()
