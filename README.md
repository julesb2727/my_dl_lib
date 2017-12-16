# my_dl_lib
This is a small deep learning library I created for my personal educational purposes. The idea was to build a small library which would allow users to easily create and train arbitrarily deep neural networks. Through designing and implementing the code without the use of higher level libraries such as PyTorch or Keras I worked through all the details involved in creating simple deep learning models. In conjunction with writing this code I was taking a Stanford online course (http://cs231n.stanford.edu/) from which I learned certain specific implementations.

# Features
I attempted to implement many state of the art features such as:
* Batch Normalization
* Dropout
* RSM-prop
* ADAM

In addition I attempted to create a fairly flexible and modular design which allows for arbitrarily sized layers and depths.
One could easily extend this library with various activations functions, different function layers, and new optimization algorithms without changing the basic structure of the code.
