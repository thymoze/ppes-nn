#include "relu.hpp"

#include <autograd/autograd.hpp>
#include <matrix/matrix.hpp>

namespace nn {

ReLU::ReLU() {}

std::vector<Variable<double>> ReLU::forward(const std::vector<Variable<double>>& inputs) {
  auto x = ag::max(inputs[0], 0.);

  return {x};
}

}  // namespace nn
