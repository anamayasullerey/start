import numpy as np
import mnist.utils.load_mnist as load_mnist
import start.neural_network as nn
import start.layer_dict as ld
import start.weight_update_params as wup

# Load the training, validation and test data
# Each data is a numpy array of shape Number of Samples * 795
# 0:783 are inputs, 784:793 are outputs, 794 is classified output
# N is chose as first dimention as it is easy to shuffle training data
# during training
training_data, validation_data, test_data = load_mnist.load_mnist()

validation_x = np.transpose(validation_data[:, 0:784]) 
validation_y_class = np.transpose(validation_data[:, 794])
val_acc = lambda: net.classification_accuracy(validation_x, validation_y_class)

test_x = np.transpose(test_data[:, 0:784]) 
test_y_class = np.transpose(test_data[:, 794])
test_acc = lambda: net.classification_accuracy(test_x, test_y_class)

##

# Creating Neural Network
print("Creating neural network")
print()
# Step 1: Create Network - specify input layer neurons (28x28=784)
net = nn.NeuralNetwork("test_net", 784)

# Step 2: Add hidden layers in sequence

# Fully connected layer
layer = ld.hdict["fc"](800)
net.add_layer(layer)

# Relu activation layer
layer = ld.hdict["relu"](800)
net.add_layer(layer)

layer = ld.hdict["fc"](10)
net.add_layer(layer)

# Add output layer
layer = ld.odict["softmax"](10)
net.add_layer(layer)

#  Neural Network definition done
net.check_arch()

# Specify l2 loss
net.set_l2_loss_coeff(.001)

# Define weight update method
params = wup.GradientDescentParams(.3)
# params = wup.MomentumParams(.3)
# params = wup.AdamParams()
net.set_weight_update_function(params)

# For repeatability during testing
# np.random.seed(1)
# Initialize the network
net.initialize_parameters()

# Set training related parameters
mini_batch_size = 32
epochs = 20
verbose = 0

print ("Starting training")
print()
# Train the network
for epoch in range(1, epochs+1):
    print("Epoch " + str(epoch))
    np.random.shuffle(training_data)
    mini_batches = [training_data[k:k + mini_batch_size, :] for k in
                   range(0, len(training_data), mini_batch_size)]
    for count, mini_batch in enumerate(mini_batches, start=1):
        x = np.transpose(mini_batch[:, 0:784])
        y = np.transpose(mini_batch[:, 784:794])
        net.train(x, y)
        if ((count%100 == 0) and verbose):
            print("Count {0} validation data accuracy = {1} %.".format(count, val_acc()))
            print()
            
        
    print("Epoch {0} validation data accuracy = {1} %.".format(epoch, val_acc()))
    print()

print("Test data accuracy = {0} %.".format(test_acc()))
print()
