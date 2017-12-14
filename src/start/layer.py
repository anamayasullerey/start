"""
Layer base class functionality.

Notes:
    
o Layer needs to have all its parametes listed in "parameters".

o The following rules need to be observed while declaring parameters in any 
  exteded class:
  1. The parameter name in the parameters list should exactly match the
     variable name.
  2. The derivatibe of the parameter is held in the variable with same name 
     with character "d" appended at the front
           
o l2_loss_coefficient is assumed generic enough for the purpose of this 
  exercise that it gets a place in the class       
"""
class Layer(object):

    def __init__(self):
        self.layer_num = -1
        self.l2_loss_coeff = 0
        self.parameters = []

     
    def check(self):
        pass
    
    """
    Forward propagation related calculations.
    Activation function needs to be defined in the extended class.
    """
    def forward_calc(self, x):
        pass

    def forward_prop(self, x):
        self.forward_calc(x)
        if self.next_layer is not None:
            self.next_layer.forward_prop(self.activations)

    def backward_calc(self):
        pass

    def backward_grad(self):
        pass

    def backward_prop(self):
        self.backward_calc()
        if (self.layer_num > 1):
            self.backward_grad ()
            self.prev_layer.backward_prop()

    def initialize_parameters(self):
        pass

    def print_forward(self):
        print("Layer number: " + str(self.layer_num))
        print("activations:")
        print(self.activations)

    def print_backward(self):
        print("Layer number: " + str(self.layer_num))
        if (self.layer_num > 0):
            print("dactivations:")
            print(self.dactivations)

    def set_l2_loss_coeff(self, l2_loss_coeff):
        self.l2_loss_coeff = l2_loss_coeff

    def get_l2_loss(self):
        return 0

    def check_num_neurons(self):
        error_str = "Error in layer {0}".format(self.layer_num) 
        error_str = "Number of neurons in " + type(self).layer_name + " " + type(self).layer_type
        error_str += " layer does not match the number of neurons in previous layer"
        assert self.prev_layer.num_neurons == self.num_neurons,  error_str
