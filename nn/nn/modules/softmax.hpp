#pragma once

#include <iostream>
#include <nn/module.hpp>
#include <vector>

namespace nn {

template <typename T>
class Softmax : public Module<T> {
 public:
  Softmax(std::optional<std::size_t> dim = std::nullopt) : dim_(dim){};

  Tensor<T> forward(const Tensor<T>& input) override { return softmax(input, dim_); };

  std::string save(const std::string&) override { return "nn::Softmax<T>()"; };

 private:
  std::optional<std::size_t> dim_;
};

}  // namespace nn
