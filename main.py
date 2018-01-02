import numpy as np
from model import model
from trainer import trainer
from keras.datasets import mnist

# create lists for 7 layer neural network
layer_list = ["affine-relu","affine-relu","batchnorm","affine-relu","affine-relu","batchnorm","affine"]
layer_dim = [(28*28,150),(150,150),(150,150),(150,150),(150,150),(150,150),(150,10)]
loss_type = "softmax"

# load mnist data using keras package
(X_train, y_train), (X_test, y_test) = mnist.load_data()
data = {}
data["X_train"] = X_train[:50000]
data["y_train"] = y_train[:50000]
data["X_val"] = X_train[50000:]
data["y_val"] = y_train[50000:]

# init and train network on mnist data, should reach close to 97% acc on test set
learning_rate = 0.00652
dropout_rate = 0.22386

my_model = model(layer_list, layer_dim, loss_type, update_rule="adam", reg=1e-4, dropout=dropout_rate)
my_trainer = trainer(my_model, data, learning_rate, num_epochs=10)
best_model = my_trainer.train()
test_rate = best_model.test(X_test,y_test)
print("test acc: ", test_rate)

# high = 1e-2
# low = 0
# num_test = 10
# for i in range(num_test):
#
#     cur_reg = round(np.random.uniform(low,high),5)
#     print("TESTING: ", best_dropout)
#     my_model = model(layer_list, layer_dim, loss_type, update_rule="adam", reg=cur_reg, dropout=best_dropout)
#     my_trainer = trainer(my_model, data, 0.00652, num_epochs=3)
#     my_trainer.train()
#     if (my_trainer.val_history[-1] > best_val):
#         best_val = my_trainer.val_history[-1]
#         best_dropout = cur_reg
#
#     print("THE BEST: ", best_val, best_dropout)
#
# #
# best_model = my_trainer.train()
# test_err = best_model.test(X_test,y_test)
# print("TEST_ERR: ", test_err)
##TODO: add another use case