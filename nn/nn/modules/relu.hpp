#pragma once

#include <nn/module.hpp>
#include <string>

namespace nn {

template <typename T>
class ReLU : public Module<T> {
 public:
  ReLU() = default;

  Tensor<T> forward(const Tensor<T>& input) override { return relu(input); };

  std::string save(const std::string&) override { return "nn::ReLU<T>()"; };

 private:
};

}  // namespace nn
