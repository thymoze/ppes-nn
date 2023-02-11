#pragma once

#include <autograd/autograd.hpp>
#include <matrix/matrix.hpp>
#include <nn/module.hpp>
#include <string>

namespace nn {

template <typename T>
class ReLU : public Module<T> {
 public:
  ReLU() = default;

  std::vector<Variable<T>> forward(const std::vector<Variable<T>>& inputs) override {
    auto x = max(inputs[0], 0.);

    return {x};
  };

  std::string save(const std::string&) override { return "nn::ReLU<T>()"; };

  bool is_prunable() override { return false; }

  void prune_one_neuron() override {}

 private:
};

}  // namespace nn
