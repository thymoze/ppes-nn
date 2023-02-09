#include <iostream>
#include <string>
#include <utility>
#include <nn/all.hpp>

#include "../trained_models/xor.hpp"

int main(void) {
  auto model = nn::Sequential<double>();
  model.add(nn::Linear<double, 2, 3>());
  model.add(nn::Sigmoid<double>());
  model.add(nn::Linear<double, 3, 1>());
  model.init(xor_model::xor_model, xor_model::xor_model_len);

  while (1) {
    std::string input;
    std::cout << "1. Input: ";
    std::getline(std::cin, input);
    float first = std::stoi(input);

    std::cout << "2. Input: ";
    std::getline(std::cin, input);
    float second = std::stoi(input);

    auto in = Tensor<double>::make({1, 2}, {first, second});
    auto out = model(in);

    std::cout << "Output: " << out.item() << std::endl;
  }
}
