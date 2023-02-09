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

  void init() override {}

  unsigned int init(const unsigned char data[], const unsigned int data_len) override { return 0; }

 private:
  std::optional<std::size_t> dim_;
};

}  // namespace nn
