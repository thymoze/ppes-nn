# Basics

Implement a barebones neural network library which allows the user to:

* compose networks from different types of layers
* train these networks through backpropagation with stochastic gradient descent

To verify the functionality of the library implement a simple non-linear example problem, for example train the network to learn XOR.

Next enable running inference on the Raspberry Pi Pico. This includes a way to serialize the trained network weights, flashing them to the Pico and restoring them, so that the device can run pre-trained networks.

# MNIST

With the basic functionality in place we can turn to more interesting problems: Implementing a network for recognizing handwritten digits from the MNIST database.

This requires parsing the dataset files available from [1], including a way to train multiple batches of images at once to speed up the learning process.

Again inference on the trained network should also run on the Pico.


[1] http://yann.lecun.com/exdb/mnist/

# Paths forward

Exploring different approaches to reducing the size of the trained network to make better use the Pico's restrictive capabilities:

* Quantization of floating-point weights into smaller datatypes like int8
* Pruning of neurons or connections
* etc.

Performance optimizations: Implement multi-threaded operations on x86, make use of multicore on the Pico and explore the usage of SIMD vector instructions like ARM Neon on the Raspberry Pi 4

Increasing the problem size: How large can you go on a device like the Pico?
