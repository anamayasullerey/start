"""
layer dictionries
"""
from . import fc_layer_h
from . import relu_layer_h
from . import sigmoid_layer_h
from . import tanh_layer_h

from . import loss_layer_o
from . import sigmoid_layer_o
from . import softmax_layer_o

hdict = {}

hdict[fc_layer_h.FcLayer.layer_name] = fc_layer_h.FcLayer
hdict[relu_layer_h.ReluLayer.layer_name] = relu_layer_h.ReluLayer
hdict[sigmoid_layer_h.SigmoidLayer.layer_name] = sigmoid_layer_h.SigmoidLayer
hdict[tanh_layer_h.TanhLayer.layer_name] = tanh_layer_h.TanhLayer

odict = {}

odict[loss_layer_o.LossLayer.layer_name] = loss_layer_o.LossLayer
odict[sigmoid_layer_o.SigmoidLayer.layer_name] = sigmoid_layer_o.SigmoidLayer
odict[softmax_layer_o.SoftmaxLayer.layer_name] = softmax_layer_o.SoftmaxLayer