
* Barebones Neural Network library:
    * Compose networks from different feed-forward and/or convolutional layers
    * Training with backpropagation on x86
    * Inference on x86/Pi4/Pico
        * Weights on Pico? Part of binary or send at runtime?

* To start: basic MNIST digit recognition network (probably just feed-forward, alternatively convolutional)

* Explore:
    * Different approaches for reducing network size (smaller datatypes, quantization, etc)
    * Increasing problem size: How big can you go on more restricted hardware like the Pico
    * Optimizing performance
        * Explore using SIMD vector instructions like ARM Neon
    * Training on Pi4 (or even Pico?)
