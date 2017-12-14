import os
import gzip
import _pickle
import wget

import numpy as np


def load_mnist():
    if not os.path.exists(os.path.join(os.curdir, 'data')):
        print ("Downloading MNIST data")
        os.mkdir(os.path.join(os.curdir, 'data'))
        wget.download('http://deeplearning.net/data/mnist/mnist.pkl.gz', out='data')
        print ()
        print ()
        print ()

    data_file = gzip.open(os.path.join(os.curdir, 'data', 'mnist.pkl.gz'), 'rb')
    training_data, validation_data, test_data = _pickle.load(data_file, encoding='iso-8859-1')
    data_file.close()
    
    res_training_data = massage_data(training_data)
    res_validation_data = massage_data(validation_data)
    res_test_data = massage_data(test_data)

    return res_training_data, res_validation_data, res_test_data

def massage_data(data):
    num_samples = data[1].shape[0]
    odata_class = data[1].reshape((num_samples,1))
    odata_onehot = np.zeros(shape=(num_samples, 10))
    odata_onehot[np.arange(num_samples), data[1]] = 1
    odata_onehot = odata_onehot.astype(float)
    result = np.append(data[0], odata_onehot, axis=1)
    result = np.append(result, odata_class, axis=1)
    return result
