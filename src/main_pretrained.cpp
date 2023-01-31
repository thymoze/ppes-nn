#include <iostream>
#include <string>
#include <utility>

#include "../trained_models/XOR_Model.hpp"

int main(void) {
  auto model = XOR_Model<float>::create();
  while (1) {
    std::string input;
    std::cout << "1. Input: ";
    std::getline(std::cin, input);
    float first = std::stoi(input);

    std::cout << "2. Input: ";
    std::getline(std::cin, input);
    float second = std::stoi(input);

    auto in = nn::Variable<float>(nn::Matrix<float>{1, 2, {first, second}});
    auto out = model({in})[0];

    std::cout << "Output: " << out(0, 0) << std::endl;
  }
}
