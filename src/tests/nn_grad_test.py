import unittest

class NnGradTest(unittest.TestCase):
        
    def define_nn(self):
        pass
    
    def set_training_example(self):
        pass
    
    def test(self):
        self.define_nn();
        self.set_training_example();
        self.assertTrue(self.net.check_gradient(self.x, self.y));
        self.assertTrue(self.extra_checks())

    def check_gradient(self):
        return self.net.check_gradient(self.x, self.y);
        
    def extra_checks(self):
        return True
    
if __name__ == '__main__':
    unittest.main()
