#pragma once

#include <autograd/autograd.hpp>
#include <iostream>
#include <matrix/matrix.hpp>
#include <nn/module.hpp>
#include <vector>

namespace nn {

template <typename T>
class Sigmoid : public Module<T> {
 public:
  Sigmoid() = default;

  std::vector<Variable<T>> forward(const std::vector<Variable<T>>& inputs) override {
    auto ones = Variable<T>(Matrix<T>(inputs[0].rows(), inputs[0].cols(), 1));
    auto x = reciprocal(ones + exp(negate(inputs[0])));

    return {x};
  };

  std::string save(const std::string&) override { return "nn::Sigmoid<T>()"; };

  bool is_prunable() override { return false; }

  void prune_one_neuron() override {}

 private:
};

}  // namespace nn
