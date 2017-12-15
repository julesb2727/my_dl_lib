import logging
import numpy as np
from updater import *

# A layer objects had three dicts, params, grads, and cache. The params are initialized on init of the layer
# and will be updated by the updater during a call to update_layer(). The grads are initialized and updated
# during a call to step backwards. The grads are passed to the updater in order to compute the update on the params.
# The cache is used to store any intermediary values what might be used later usually during step_backward().
# Layers and their functions do not need to be initialized or called directly. This is handled by the model class.

class affine_layer(object):

    def __init__(self, dims, weight_scale, update_rule, reg, dropout, id):

        input_dim, hidden_dim = dims
        self.id = id # for debuging purposes

        # after a successful step_forward and step_backward a layer should have values
        # for its params and the corresponding grads
        self.params = {}
        self.grads = {}
        self.cache= {}

        self.reg = reg
        self.dropout = dropout

        self.params["W"] = np.random.randn(input_dim,hidden_dim) * weight_scale
        self.params["b"] = np.zeros(hidden_dim)

        # each layer contains its own updater, can pass params for more complex update rules
        # which require know the shape of params in order to keep track of some update stats.
        self.updater = updater(update_rule, params=self.params)

    def step_forward(self, x, mode="train"):

        w = self.params["W"]
        b = self.params["b"]
        self.cache["x"] = x


        out = x.reshape(x.shape[0], -1)
        out = out.dot(w) + b

        if(self.dropout > 0 and mode == "train"):
            drop_mask = (np.random.rand(*out.shape) < (1-self.dropout)) / (1-self.dropout) # inverted dropout
            self.cache["d_mask"] = drop_mask
            out *= drop_mask

        return out

    # calculate and update the grad params for layer given an upstream gradient
    def step_backward(self, upstream_grad):

        w = self.params["W"]
        b = self.params["b"]
        x =  self.cache["x"]

        if (self.dropout > 0):
            upstream_grad *= self.cache["d_mask"]

        self.grads["W"] = x.reshape(x.shape[0], -1).T.dot(upstream_grad) + (self.reg * w)
        dx = upstream_grad.dot(w.T).reshape(x.shape)
        self.grads["b"] = np.sum(upstream_grad, axis=0)

        return dx

    def update_layer(self, learning_rate):
        self.params = self.updater.update_params(self.params, self.grads, learning_rate)

# activation layer meant to follow an affine layer. Capable of computing multiple types of activation.
class activation_layer(object):

    def __init__(self, type, id):

        self.params = {}
        self.grads = {}
        self.cache = {}
        self.type = type
        self.updater = None
        self.id = id

    def step_forward(self, x, mode="train"):

        self.cache["x"] = x
        out = x

        if self.type == "relu":
            out = np.maximum(0, x)
        else:
            logging.warning('activation type %s not found, passing x forward unchanged', self.type)

        return out

    # calculate and update the grads for layer given an upstream gradient
    def step_backward(self, upstream_grad):

        x = self.cache["x"]
        dx = upstream_grad

        if self.type == "relu":
            dx[x <= 0] = 0

        return dx

    def update_layer(self, learning_rate):
        pass



class batch_norm_layer(object):

    def __init__(self, dims, update_rule, id, **kwargs):

        input_dim, hidden_dim = dims
        self.id = id

        self.params = {}
        self.grads = {}
        self.cache = {}

        self.momentum = kwargs.pop('momentum', 0.9)
        self.eps = kwargs.pop('eps', 1e-5)

        self.running_mean = np.zeros((hidden_dim))
        self.running_var = np.zeros((hidden_dim))

        self.params["gamma"] = np.ones((hidden_dim))
        self.params["beta"] = np.zeros((hidden_dim))

        # each layer contains its own updater
        self.updater = updater(update_rule, params=self.params)


    def step_forward(self, x, mode="train"):

        if mode=="train":

            mu = np.mean(x, axis=0)
            # Variance
            var = 1 / float(x.shape[0]) * np.sum((x - mu) ** 2, axis=0)
            # Normalized Data
            x_hat = (x - mu) / np.sqrt(var + self.eps)
            # Scale and Shift
            y = self.params["gamma"] * x_hat + self.params["beta"]
            out = y

            # Make the record of means and variances in running parameters
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self.cache["x"] = x
            self.cache["mu"] = mu
            self.cache["var"] = var

        elif mode=="test":

            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            # Scale and Shift
            y = self.params["gamma"] * x_hat + self.params["beta"]
            out = y

        else:
            logging.warning('invalid batchnorm mode: %s', mode)

        return out

    def step_backward(self, upstream_grad):

        gamma = self.params["gamma"]
        x = self.cache["x"]
        mu = self.cache["mu"]
        var = self.cache["var"]
        N = upstream_grad.shape[0]

        self.grads["beta"] = np.sum(upstream_grad, axis=0)
        self.grads["gamma"] = np.sum((x - mu) * (var + self.eps) ** (-1. / 2.) * upstream_grad, axis=0)
        dx = (1. / N) * gamma * (var + self.eps) ** (-1. / 2.) * (N * upstream_grad - np.sum(upstream_grad, axis=0) - (x - mu) * (var + self.eps) ** (-1.0) * np.sum(upstream_grad * (x - mu), axis=0))

        return dx

    def update_layer(self, learning_rate):
        self.params = self.updater.update_params(self.params, self.grads, learning_rate)

def softmax_loss(x, y, weights, reg):

        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = x.shape[0]
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N


        reg_loss = 0
        for w in weights:
            reg_loss += np.sum(w*w)

        loss += reg_loss * reg * 0.5

        return dx, loss


