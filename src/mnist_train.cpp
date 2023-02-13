#define DEFAULT_TENSOR_BACKEND TensorBackend<T>(MTOps<T>())

#include <algorithm>
#include <iostream>
#include <nn/all.hpp>
#include <nn/dataset/mnist_dataset.hpp>
#include <nn/quantization/quantization.hpp>
#include <vector>

#include "train.hpp"

using tensor::Tensor;

int main() {
  nn::random::seed(0x5EED);

  auto model = nn::Sequential<float>();
  model.add(nn::Linear<float>(28 * 28, 200));
  model.add(nn::Sigmoid<float>());
  model.add(nn::Linear<float>(200, 10));
  model.init();

  model.save("../../trained_models/mnist_float_200.hpp", "mnist_model");

  train(model, 100, 1, 0.1);
  model.save("../../trained_models/mnist_float_200.hpp", "mnist_model");

  evaluate(model);
}
