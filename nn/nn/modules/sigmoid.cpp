#include "sigmoid.hpp"

#include <autograd/autograd.hpp>
#include <iostream>
#include <matrix/matrix.hpp>

namespace nn {

Sigmoid::Sigmoid() {}

std::vector<Variable<double>> Sigmoid::forward(const std::vector<Variable<double>>& inputs) {
  auto ones = Variable<double>(Matrix<double>(inputs[0].rows(), inputs[0].cols(), 1));
  auto x = ag::reciprocal(ones + ag::exp(ag::negate(inputs[0])));

  return {x};
}

std::string Sigmoid::save(std::string model_name) {
    return "nn::Sigmoid()";
}

}  // namespace nn
