#pragma once

#include <autograd/autograd.hpp>
#include <matrix/matrix.hpp>

namespace nn {
template <typename T>
Variable<T> mse(Variable<T> pred, Variable<T> target) {
  auto diff = target - pred;
  return mean(diff * diff);
}

}  // namespace nn
