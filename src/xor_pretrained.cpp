#include <iostream>
#include <nn/all.hpp>
#include <string>
#include <utility>

#include "../trained_models/xor.hpp"

using tensor::Tensor;

int main(void) {
  auto model = nn::Sequential<float>();
  model.add(nn::Linear<float>(2, 3));
  model.add(nn::Sigmoid<float>());
  model.add(nn::Linear<float>(3, 1));
  model.init(xor_model::xor_model, xor_model::xor_model_len);

  while (1) {
    std::string input;
    std::cout << "1. Input: ";
    std::getline(std::cin, input);
    float first = std::stoi(input);

    std::cout << "2. Input: ";
    std::getline(std::cin, input);
    float second = std::stoi(input);

    auto in = tensor::make<float>({1, 2}, {first, second});
    auto out = model(in);

    std::cout << "Output: " << out(0, 0) << std::endl;
  }
}
