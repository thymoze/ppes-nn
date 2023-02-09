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

  void init() override {}

  unsigned int init(const unsigned char data[], const unsigned int data_len) override { return 0; }

 private:
};

}  // namespace nn
