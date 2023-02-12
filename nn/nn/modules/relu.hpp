#pragma once

#include <nn/module.hpp>
#include <string>

namespace nn {

using namespace tensor;

template <typename T>
class ReLU : public Module<T> {
 public:
  ReLU() = default;

  Tensor<T> forward(const Tensor<T>& input) override { return relu(input); };

  void init() override {}

  unsigned int init(const unsigned char data[], const unsigned int data_len) override { return 0; }

  bool is_prunable() override { return false; }

  int prune_one_neuron() override { return -1; }
  void apply_pruned_neuron(int) override {}
  bool is_linear() override { return false; }

  std::vector<std::uint8_t> data() override { return {}; }

 private:
};

}  // namespace nn
