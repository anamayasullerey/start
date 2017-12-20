# *Start* - A beginner's neural net framework

**_Start_** is designed for beginner level AI learners to experiment with neural network concepts. It is inspired by and covers the concepts covered in first three courses of the deep learning specialization by [Prof Andrew Ng](http://www.andrewng.org/ )  ([deeplearning.ai](https://www.deeplearning.ai/)).

# How to *Start*
After downloading the git repository for *Start*, **"start/src"** directory needs to be added to **$PYTHONPATH**.

## 1. [MNIST digit classifier example](src/mnist/mnist_net.ipynb) 
MNIST digit classfier example using *Start* gives a feel of the architecture of *Start*. It is recommended to follow the documentation and create a classifier in Jupyter notebook. The example, **mnist_net.py** is also checked in as python module and can be run in a shell.

## 2. [Start documentation](start_manual.md)
The documentation on Start explains the featrues of the *Start*. It also covers the important functions of the API.

# Modifying *Start*
## Architecture of *Start*
*Start* builds a network as an array of layers. There layers have handles to their previous and next layers. Forward propagation and backward propagation is done as a recursion across the layers. Weight update functionality is separated from the layers. The state for weight update is maintained within the layers.

## Unit testing
*Start* comes with unit tests. These tests are done on very small scale nets which can be debugged manually. It is **highly recommended** to add unit tests for any new features. The unit tests that come with the framework are divided into two categories.

1. **Gradient Tests** : These tests check the gradients of various combinations of layers and cost functs
2. **Training Tests** : These tests train the network and check the preditions of the network. Training data for these tests is generated by a model function.

The tests can be run with the following command in the tests directory
```
python -m unittest discover -v
```
