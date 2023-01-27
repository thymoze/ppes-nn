#pragma once

#include <autograd/autograd.hpp>
#include <matrix/matrix.hpp>

namespace nn {

Variable<double> mse(Variable<double> pred, Variable<double> target) {
  return mean((target - pred) * (target - pred));
}

}  // namespace nn
