#pragma once

#include <tensor/tensor.hpp>

namespace nn {

template <typename T>
Tensor<T> mse(const Tensor<T>& pred, const Tensor<T>& target) {
  auto diff = target - pred;
  return mean(diff * diff);
}

}  // namespace nn
