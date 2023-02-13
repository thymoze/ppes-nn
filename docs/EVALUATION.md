# Evaluation

This document contains comparisons of inference time, binary size and accuracy of models of different sizes and data types on the Pico.

# Double vs Float

As a starting point we will compare a model with one hidden layer of size 200 using 64-bit and 32-bit weights. The model was trained on the MNIST training set for 1 epoch with a batch size of 100 and a learning rate of 0.1. 

Note that due to RAM-constraints† we will only evaluate the accuracy on the first 20 images of the test set on the Pico. The accuracy on the full 10k test-set was evaluated on x86.

|                                             | `double` | `float` |
| ------------------------------------------- | -------- | ------- |
| Accuracy on full test set (x86)             | 90.17%   | 90.16%  |
| Binary size                                 | 1.67 MB  | 1.03 MB |
| Accuracy on first 20 images of the test set | 95%      | 95%     |
| Inference time                              | 24.85s   | 19.79s  |
| Time per image                              | 1.24s    | 0.99s   |

We can observe that using 32-bit floating point numbers does not impact accuracy in any negative way compared to 64-bit. It does however speed up inference time by ~25%. From this point forward all experiment will be run only with 32-bit precision floating-point numbers.

<small>
† In the current implementation the entire dataset is kept in RAM. The 264kB RAM of the Pico could at most fit 264 kB / (28 * 28 * 8 B/image) = 42 images with double-precision. To leave some room for the rest of the program this test only used 20. Later evaluations using only float-precision will use 50 images.
</small>

# Different sizes

In this test we will compare networks of different sizes. All networks will be trained for 1 epoch using a batch size of 100 and a learning rate of 0.1. A heading of "(N)" means a single hidden layer of size N was used, "(N, M)" means 2 hidden layers of sizes N and M were used.

|                                             | (50)   | (100)  | (200)   | (300)   | (300,100) |
| ------------------------------------------- | ------ | ------ | ------- | ------- | --------- |
| Accuracy on full test set (x86)             | 86.11% | 89.02% | 90.16%  | 91.58%  | 84.89%    |
| Binary size                                 | 579kB  | 738kB  | 1.03 MB | 1.37 MB | 1.49 MB   |
| Accuracy on first 50 images of the test set | 88%    | 94%    | 94%     | 94%     | 84%       |
| Inference time                              | 12.6s  | 24.89s | 49.47s  | 74.05s  | 83.71s    |
| Time per image                              | 0.25s  | 0.50s  | 0.99s   | 1.48s   | 1.67s     |

It is clearly visible that a deeper network with multiple hidden layers is not helpful for this problem, as performance goes down across the board.  
The other tests show a fairly linear relationship in inference time increasing with layer size, while accuracy is pretty close among all of them.

# Pruning

One technique to reduce the size of a network is to "prune" the least relevant neurons, removing them completely from the network and thus reducing the size of the weight matrices.  
In this test we prune a network with 300 hidden neurons which was trained for 2 epochs down to 100 neurons in 2 steps, training for an additional epoch after each prune. Like before all training was with a batch size of 100 and a learning rate of 0.1.

|                                             | Original | Pruned |
| ------------------------------------------- | -------- | ------ |
| Accuracy on full test set (x86)             | 92.97%   | 91.89% |
| Binary size                                 | 1.37 MB  | 737 kB |
| Accuracy on first 50 images of the test set | 96%      | 94%    |
| Inference time                              | 73.66s   | 24.77s |
| Time per image                              | 1.47s    | 0.50s  |

In our tests pruning was really effective at reducing the network size and as a result inference time while only losing a marginal amount of accuracy.

# Quantization

Another approach to reducing network size is quantization. Instead of storing weights data as 32-bit floating points they are stored as 8-bit integers. This can also speed up performance as the matrix multiplications can be performed on the integers instead of the floats.  
We quantize the 300 hidden neuron network from the previous test:

|                                             | Original | Quantized |
| ------------------------------------------- | -------- | --------- |
| Accuracy on full test set (x86)             | 92.97%   | 10.81%    |
| Binary size                                 | 1.37 MB  | 689 kB    |
| Accuracy on first 50 images of the test set | 96%      | 12%       |
| Inference time                              | 73.66s   | 73.45s    |
| Time per image                              | 1.47s    | 1.47s     |

Not only did quantizing completely obliterate the accuracy of the network but it was also not significantly faster to calculate. Only the binary size is indeed significantly smaller.  
It is likely that there are errors in our implementation of quantization as these results are no better than random chance.