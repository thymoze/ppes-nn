#pragma once

#include <nn/module.hpp>
#include <string>

namespace nn {

template <typename T>
class ReLU : public Module<T> {
 public:
  ReLU() = default;

  Tensor<T> forward(const Tensor<T>& input) override { return relu(input); };

  void init() override {}

  unsigned int init(const unsigned char data[], const unsigned int data_len) override { return 0; }

 private:
};

}  // namespace nn
