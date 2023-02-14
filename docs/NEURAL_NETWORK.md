# Neural Network library

## Modules

The API of the library is heavily inspired by PyTorch. Networks are composed of `Modules` which implement different layers and activation functions, e.g. `Linear` for a simple feed-forward layer or `Sigmoid` for a module applying the sigmoid activation function.

The basic building block for the simple networks we built is the `Sequential` module container. Modules can be added to the container using the `add()` function. When called, this module will then forward its input to the first contained module and chain the outputs to inputs sequentially for each contained module, returning the output of the final one.

```cpp
auto model = nn::Sequential<float>();
model.add(nn::Linear<float>(10, 20));
model.add(nn::Sigmoid<float>());
model.init()
```

After creating a model, it first needs to be initialized before it can be used. Calling `model.init()` will randomly initialize the weights of each layer. Alternatively one can pass the data array, written by calling `model.save()` to initialize the model from existing weights.

## Training

Now that the network is initialized we can start to train it. For that we need to define an optimization function like SGD (stochastic gradient descent). This will take care of updating the weights, each time the `step()` function is called. To know how to update the weights we also need a loss-metric, e.g. MSE (mean-squared error). Calling `backward()` on the loss value will utilize the automatic differentiation functionality, built into the `Tensor`s to backpropagate the error through the network and calculate the derivatives of the weights.

```cpp
auto optimizer = nn::SGD<float>(model.params(), learning_rate);

// Forward pass
auto output = model(input);

// Calculate the error
auto loss = nn::mse(output, target);

// Backward pass and weights update
loss.backward();
optimizer.step();

// Reset gradients for next training step
optimizer.zero_grad();
```
