import numpy as np
from layers import *

# The model object initializes holds a list of all the layers of the network.
# It also provides functions which govern the flow of parameters through the network
# and can compute a forward, backward, or optimaztion pass through the entire
# network.

class model(object):

    """
    -------Inputs-------
    layer_list: a list of strings giving the sequence of layer types, e.g. ["affine-relu","affine"]
    layer_dims: a list of tuples giving the dimensions of each layer, e.g. [(256,100),(100,10)]
    loss_type: a string indicating the type of loss function for model, e.g. "softmax"
    update_rule: string giving the update rule which the model will perform during optimization
    weight_scale: scalar giving the standard deviation for random initialization of the weights.
    reg: scalar giving L2 regularization strength.
    dropout: scalar giving the probability of a node dropping out on a forward pass
    """

    def __init__(self, layer_list, layer_dims, loss_type, update_rule="sgd", weight_scale=1e-3, reg=0.0, dropout=0.0):

        self.layers = []
        self.loss_type = loss_type
        self.reg = reg
        self.dropout = dropout

        for i in range(len(layer_list)):

            layer_name = layer_list[i]
            layer_dim = layer_dims[i]

            if "-" in layer_name:
                aff = affine_layer(layer_dim, weight_scale, update_rule, reg, dropout, "a" + str(i))
                acti_type = layer_name[(layer_name.index("-")+1):]
                acti = activation_layer(acti_type, "r"+str(i))
                self.layers.append(aff)
                self.layers.append(acti)

            elif layer_name == "affine":
                cur = affine_layer(layer_dim, weight_scale, update_rule, reg, dropout, "a"+str(i))
                self.layers.append(cur)

            elif layer_name == "batchnorm":
                cur = batch_norm_layer(layer_dim, update_rule, "bn"+str(i))
                self.layers.append(cur)

            elif layer_name == "relu":
                cur = activation_layer("relu", "r"+str(i))
                self.layers.append(cur)

            else:
                logging.warning('%s layer type not found, layer not initialized', layer_name)

        # remove dropout on last layer
        self.layers[-1].dropout = 0

    # perform a forward pass through the network and return activations of last layer and weights for regularization
    def forward_pass(self, x, mode="train"):

        layer_in = x
        weights = []

        for layer in self.layers:
            # gather weights for regularization
            if 'W' in layer.params:
                weights.append(layer.params["W"])
            layer_out = layer.step_forward(layer_in,mode)
            layer_in = layer_out

        return layer_out, weights

    # x here should be the last layer of activations. The weights of the network are passed in order to computer regularization
    def calculate_loss(self, x, y, weights):

        if self.loss_type == "softmax":
            return softmax_loss(x,y,weights,self.reg)

    # perform a backward pass through the network, updating the grad params of each layer along the way
    def backward_pass(self, dx):

        upstream_grad = dx

        for layer in reversed(self.layers):
            downstream_grad = layer.step_backward(upstream_grad)
            upstream_grad = downstream_grad

    def optimize(self, learning_rate):
        for layer in self.layers:
            layer.update_layer(learning_rate)

    def test(self, x, y):

        layer_out, weights = self.forward_pass(x, mode="test")
        prediction = np.argmax(layer_out, axis=1)
        test_rate = np.sum(prediction == y) / prediction.shape[0]


        return test_rate
