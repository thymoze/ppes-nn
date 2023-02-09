#pragma once

#include <algorithm>
#include <tensor/tensor.hpp>
#include <vector>

namespace nn {

template <typename T>
class SGD {
 public:
  SGD(const std::vector<Parameter<T>>& parameters, double learning_rate = 0.01)
      : params_(parameters), learning_rate_(tensor::make<T>(learning_rate)) {}

  void step() {
    for (auto& param : params_) {
      auto& data = param.value();
      assert(data.grad() && "Parameter must have a gradient.");
      auto& grad = *data.grad();

      param.update(data - (learning_rate_ * grad));
    }
  }

  void zero_grad() {
    for (auto& param : params_) {
      param.value().zero_grad();
    }
  }

 private:
  std::vector<Parameter<T>> params_;
  Tensor<T> learning_rate_;
};

}  // namespace nn
