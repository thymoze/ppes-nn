#define DEFAULT_TENSOR_BACKEND TensorBackend<T>(MTOps<T>())

#include <iostream>
#include <nn/all.hpp>
#include <nn/quantization/quantization.hpp>
#include <string>
#include <utility>

#include "../trained_models/mnist_float_300_2_pruned_100.hpp"
#include "train.hpp"

using tensor::Tensor;

int main(void) {
  auto model = nn::Sequential<float>();
  model.add(nn::quantization::QLinear<float>(28 * 28, 100));
  model.add(nn::Sigmoid<float>());
  model.add(nn::quantization::QLinear<float>(100, 10));
  model.init(mnist_model::mnist_model, mnist_model::mnist_model_len);

  evaluate<float>(model);
}
