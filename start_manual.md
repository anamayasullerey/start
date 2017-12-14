# *Start* manual

This document goes over the functionality and features of the modules in *Start*.

## 1. neural_network
This class captures architecture of the neural network. 

### 1.1 Module import
neural_network module is imported by the following statement.
```
import start.neural_network as nn

```
### 1.2 Creating an object
A neural network object is created by the following statement.
```
net = nn.NeuralNetwork("name", number_of_inputs)
```

### 1.3 Adding layers
Once created, layers are sequentially added to the network from input to output. The final layer is the output layer.
```
net.add_layer(layer)
```

### 1.4 Specifying L2 loss coefficient
```
net.set_l2_loss_coeff(l2_loss_coefficient)
```

### 1.5 Setting weight update function
```
net.set_weight_update_function(weight_update_parameters)
```

### 1.6 Sanity check
Once the network is defined a sanity check of the architecture is done by calling check_arch() function.
```
net.check_arch()
```
### 1.7 Initialization
The network needs to be initialized (random initialization) before training.
```
net.initialize_parameters()
```

### 1.8 Training
```
# x : input 2D numpy array of size (number of inputs * batch size)
# y : output 2D numpy array of size (number of outputs * batch size)
net.train(x, y) 
```

### 1.9 Predictions
```
y = net.predict(x) 
```

### 1.10 Other useful functions
```
net.forward_prop(x) # does forward propagation
net.backward_prop(y) # does backward propagation
loss = net.loss(y) # Returns loss. Called after net.forward_prop(x) or net.predict(x)
y = net.predict_classify(x) # Returns the index for classification based outputs
status = net.check_gradient(x, y) # Returns boolean. Numerically checks the gradient calculations.
net.print_state() # prints the activation, derivatives and parameters for each layer
```
## 2. Layers
Layers in *Start* is a entity that specifies forward propagation and backward propagation methods. Every layer stores activations and corresponding input derivatives (dactivations). The layers are stored in two layer dictionaries, one for hidden layers (hdict) and one for output layers (ldict). Layer dictionaries are imported by the following statement.
```
    import start.layer_dict as ld
```

### 2.1 input layer
Input layer is is automatically generated when a network instance is created. Users do not have to worry about this layer.

### 2.2 hidden layers
Listed below are the input layer types and the code to generate them.
#### 2.2.1 fully connected (y = wx + b)
```
layer = ld.hdict["fc"](number_of_neurons)
```
#### 2.2.2 relu
```
layer = ld.hdict["relu"](number_of_neurons)
```
#### 2.2.3 sigmoid
```
layer = ld.hdict["sigmoid"](number_of_neurons)
```
#### 2.2.4 tanh
```
layer = ld.hdict["tanh"](number_of_neurons)
```

**_Note that "layer" frequently represents a fully connected function followed by an activation function. In Start these are separate layers._**

### 2.3 output layers
Listed below are the output layer types and the code to generate them.
#### 2.3.1 loss
This is the generic output layer that has an identity activation function (y=x). A loss function is specified the when this output layer is created. Following loss functions are supported for loss output layer.
* sigmoid_cross_entropy_loss
* linear_mean_squared_loss
```
layer = ld.odict["loss"]("linear_mean_squared_loss")
```
#### 2.3.2 sigmoid
This layer has a sigmoid activation function as well a sigmoid cross entropy loss.
```
layer = ld.odict["sigmoid"](number_of_neurons)
```
### 2.3.3 softmax
This layer has a softmax activation function with a softmax cross entropy loss. When using this layer the outputs need to be logits (one hot binary set).
```
layer = ld.odict["softmax"](number_of_neurons)
```
## 3. Weight update functions
Weight update module is imported by the following statement.
```
import start.weight_update_params as wup
```
Weight update function is passed to the net by the follwoing api.
```
net.set_weight_update_function(weight_update_params)
```
For each weight update function their is a unique parameter class. An object of this class is passed as **_weight_update_parameter_** in the above call.

### 3.1 Gradient descent
Following code generates the weight update parametrs for gradient descent.
``` 
# defaults : learning_rate = 0.5
weight_update_params = wup.GradientDescentParams(learning_rate)

```
### 3.2 Momentum
Following code generates the weight update parametrs for momentum method.
```
# defaults : learning_rate = 0.2, beta = 0.9
weight_update_params = wup.MomentumParams(learning_rate, beta)

```
### 3.3 Adam
Following code generates the weight update parametrs for Adam.
```
# defaults : learning_rate = 0.2, beta1 = 0.9, beta2 = 0.999, epsilon=1e-8
weight_update_params = wup.AdamParams(learning_rate, beta1, beta2, epsilon)

```


