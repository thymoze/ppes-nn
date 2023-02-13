#include <pico/stdlib.h>

#define DEFAULT_TENSOR_BACKEND TensorBackend<T>(SimpleOps<T>())

#include <iostream>
#include <nn/all.hpp>
#include <string>

#include "../../trained_models/mnist_float_100.hpp"
#include "evaluate.hpp"

using tensor::Tensor;

int main(void) {
  stdio_init_all();

  auto model = nn::Sequential<float>();
  model.add(nn::Linear<float>(28 * 28, 100));
  model.add(nn::Sigmoid<float>());
  model.add(nn::Linear<float>(100, 10));
  model.init(mnist_model::mnist_model, mnist_model::mnist_model_len);
  for (auto& p : model.params()) {
    p.template value<Tensor<float>>().requires_grad(false);
  }

  sleep_ms(5000);

  std::cout << "Model initialized" << std::endl;

  evaluate<float>(&model);
}
