#pragma once

#include <iostream>
#include <nn/module.hpp>
#include <vector>

namespace nn {

using namespace tensor;

template <typename T>
class Sigmoid : public Module<T> {
 public:
  Sigmoid() = default;

  Tensor<T> forward(const Tensor<T>& input) override { return sigmoid(input); };

  void init() override {}

  unsigned int init([[maybe_unused]] const unsigned char data[],
                    [[maybe_unused]] const unsigned int data_len) override {
    return 0;
  }

  bool is_prunable() override { return false; }

  int prune_one_neuron() override { return -1; }
  void apply_pruned_neuron(int) override{};
  bool is_linear() override { return false; }

 private:
};

}  // namespace nn
