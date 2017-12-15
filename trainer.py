import numpy as np
import logging
from model import *
from copy import deepcopy

# The trainer object holds the model and the training and validation data.
# The train() function samples mini-batches from the data and calls the relevant functions
# on the model in order to make forward passes, backward passes and optimizations.
# The model is trained for a certain number of epochs and validated along the way.
# The trainer automatically keeps track of the model with the best performance on the validation
# set which the train function returns upon completion.

class trainer(object):

    """
    -------Inputs-------
    model: an initialized model.
    data: a dictionary containing the training in validation data.
    learning_rate: a scalar indicating the learning rate for the optimaztion algorithm.

    batch_size: a scalar indicating the mini-batch size.
    lr_decay: a scalar indicating learning rate decay which occurs after every epoch.
    num_epochs: a scalar indicating the number of epochs the model should be trained for.
    validate_every: a scalar indicating that validation should occur after every n mini-batches.
    print_every: a scalar indicating that the train acc, most recent val acc, and loss should be printed after every n mini-batches.
    reg: scalar giving L2 regularization strength.
    dropout: scalar giving the probability of a node dropping out on a forward pass.
    """

    def __init__(self, model, data, learning_rate, **kwargs):

        self.model = model

        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        self.learning_rate = learning_rate
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.lr_decay = kwargs.pop('lr_decay', 1.0)

        self.validate_every = kwargs.pop('validate_every', 100)
        self.print_every = kwargs.pop('print_every', 100)
        self.train_history = []
        self.val_history = []

    def train(self):

        best_val_rate = 0
        best_model = None

        for cur_epoch in range(self.num_epochs):

            print("---------------------- EPOCH: %d ----------------------" % (cur_epoch))

            # shuffle training data
            random_mask = np.random.permutation(self.X_train.shape[0])
            self.X_train = self.X_train[random_mask]
            self.y_train = self.y_train[random_mask]

            num_batches = max(self.X_train.shape[0] // self.batch_size, 1)

            # train for one epoch
            for i in range(0,num_batches):

                # sample the batch
                X_train_batch = self.X_train[i*self.batch_size:(i+1)*self.batch_size]
                y_train_batch = self.y_train[i*self.batch_size:(i+1)*self.batch_size]

                layer_out, weights = self.model.forward_pass(X_train_batch)
                dx, loss = self.model.calculate_loss(layer_out, y_train_batch, weights)
                self.model.backward_pass(dx)
                self.model.optimize(self.learning_rate)

                if (i % self.validate_every) == 0:

                    val_rate = self.model.test(self.X_val, self.y_val)

                    train_prediction = np.argmax(layer_out, axis=1)
                    train_rate = np.sum(y_train_batch == train_prediction) / self.batch_size

                    self.train_history.append(train_rate)
                    self.val_history.append(val_rate)

                    # keep track of model with best performace on val set
                    if (val_rate > best_val_rate):
                        best_val_rate = val_rate
                        best_model = deepcopy(self.model)


                if (i % self.print_every) == 0:
                    print("train acc: %f validation acc: %f  loss: %f" % (train_rate, self.val_history[-1], loss) )

            self.learning_rate *= self.lr_decay

        return best_model