#pragma once

#include <tensor/tensor.hpp>

namespace nn {

using tensor::Tensor;

template <typename T>
Tensor<T> mse(const Tensor<T>& pred, const Tensor<T>& target) {
  auto diff = target - pred;
  return tensor::mean(diff * diff);
}

}  // namespace nn
