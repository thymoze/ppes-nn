#pragma once

#include <iostream>
#include <nn/module.hpp>
#include <vector>

namespace nn {

template <typename T>
class Sigmoid : public Module<T> {
 public:
  Sigmoid() = default;

  Tensor<T> forward(const Tensor<T>& input) override { return sigmoid(input); };

  std::string save(const std::string&) override { return "nn::Sigmoid<T>()"; };

 private:
};

}  // namespace nn
