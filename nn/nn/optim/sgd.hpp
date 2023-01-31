#pragma once

#include <algorithm>
#include <autograd/autograd.hpp>
#include <vector>

namespace nn {

template <typename T>
class SGD {
 public:
  SGD(const std::vector<Variable<T>>& parameters, double learning_rate = 0.01)
      : _params(parameters), _learning_rate(learning_rate) {}

  void step() {
    for (auto& param : _params) {
      auto& grad = param.grad().value();
      auto& data = param.value();

      data = data - (_learning_rate * grad);
    }
  }

  void zero_grad() {
    for (auto& param : _params) {
      param.zero_grad();
    }
  }

 private:
  std::vector<Variable<T>> _params;
  double _learning_rate = 0;
};

}  // namespace nn
