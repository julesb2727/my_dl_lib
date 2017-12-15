import numpy as np
import logging

# An updater object can be initialized for various different update rules. The object
# stores any parameters necessary to compute an update (e.g. momentum, first and second moments, etc.).
# During a update, the updater takes the parameters, gradients, and learning rate and
# returns the updated parameters.

class updater(object):


    def __init__(self, update_rule, **kwargs):

        self.update_rule = update_rule
        params = kwargs.pop('params', None)

        if self.update_rule == "sgd":
          pass

        elif self.update_rule == "sgd_momentum":
          self.momentum = kwargs.pop('momentum', 0.9)
          self.velocity = {}
          for key in params:
              self.velocity[key] = np.zeros_like(params[key])

        elif self.update_rule == "rmsprop":
          self.rsm_decay = kwargs.pop('rsm_decay', 0.99)
          self.grads_sqrd = {}
          for key in params:
            self.grads_sqrd[key] = np.zeros_like(params[key])

        elif self.update_rule == "adam":
          self.beta1 = kwargs.pop('beta1', 0.9)
          self.beta2 = kwargs.pop('beta2', 0.999)
          self.itr = 0

          self.first_moment = {}
          self.sec_moment = {}

          for key in params:
              self.first_moment[key] = np.zeros_like(params[key])
              self.sec_moment[key] = np.zeros_like(params[key])

        else:
          logging.warning('update rule %s not found, updater not properly init', self.update_rule)


    def sgd(self, x, dx, learning_rate):

        for key in x:
          x[key] -= learning_rate  * dx[key]

        return x

    def sdg_momentum(self, x, dx, learning_rate):

      for key in x:
        self.velocity[key] = self.momentum * self.velocity[key] - (learning_rate * dx[key])
        x[key] += self.velocity[key]

      return x

    def rmsprop(self, x, dx, learning_rate):

      for key in x:
        self.grads_sqrd[key] = self.rsm_decay * self.grads_sqrd[key] + (1 - self.rsm_decay) * dx[key] * dx[key]
        x[key] -= learning_rate * dx[key] / (np.sqrt(self.grads_sqrd[key] + 1e-8))

      return x

    def adam(self, x, dx, learning_rate):

        b1 = self.beta1
        b2 = self.beta2
        self.itr += 1
        t = self.itr

        for key in x:

            #print("updating param: ",self.itr, key, x[key].shape, dx[key].shape,  self.first_moment[key].shape, self.sec_moment[key].shape)

            lr_t = learning_rate * np.sqrt(1- np.power(b2,t)) / (1 - np.power(b1,t))

            self.first_moment[key] = b1 * self.first_moment[key] + (1-b1) * dx[key]
            self.sec_moment[key] = b2 * self.sec_moment[key] + (1-b2) * dx[key]*dx[key]

            x[key] = x[key] - (lr_t * self.first_moment[key]) / (np.sqrt(self.sec_moment[key]) + 1e-8)

        return x



    def update_params(self, x, dx, learning_rate):

        if self.update_rule == "sgd":
            return self.sgd(x, dx, learning_rate)

        elif self.update_rule == "sgd_momentum":
            return self.sdg_momentum(x, dx, learning_rate)

        elif self.update_rule == "rmsprop":
            return self.rmsprop(x, dx, learning_rate)

        elif self.update_rule == "adam":
            return self.adam(x, dx, learning_rate)