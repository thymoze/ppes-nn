#pragma once

#include <autograd/autograd.hpp>
#include <matrix/matrix.hpp>

namespace nn {

Variable<double> mse(Variable<double> pred, Variable<double> target) {
  auto diff = target - pred;
  return mean(diff * diff);
}

}  // namespace nn
