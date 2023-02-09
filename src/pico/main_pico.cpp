#include <pico/stdlib.h>

#define DEFAULT_TENSOR_BACKEND TensorBackend<T>(SimpleOps<T>())

#include <iostream>
#include <nn/all.hpp>
#include <string>

#include "../../trained_models/xor.hpp"

int main(void) {
  stdio_init_all();

  auto model = nn::Sequential<double>();
  model.add(nn::Linear<double, 2, 3>());
  model.add(nn::Sigmoid<double>());
  model.add(nn::Linear<double, 3, 1>());
  model.init(xor_model::xor_model, xor_model::xor_model_len);

  while (1) {
    std::string input;
    std::cout << "1. Input: ";
    std::getline(std::cin, input, '\r');
    double first = std::stoi(input);

    std::cout << "2. Input: ";
    std::getline(std::cin, input, '\r');
    double second = std::stoi(input);

    auto in = tensor::make<double>({1, 2}, {first, second});
    auto out = model(in);

    std::cout << "Output: " << out.item() << std::endl;
  }
}
