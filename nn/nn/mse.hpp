#pragma once

#include <autograd/autograd.hpp>
#include <matrix/matrix.hpp>

Variable<double> mse(Variable<double> pred, Variable<double> target) {
  return ag::mean((target - pred) * (target - pred));
}