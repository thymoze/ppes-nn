#include "linear.hpp"

#include <algorithm>
#include <autograd/autograd.hpp>
#include <matrix/matrix.hpp>
#include <random>

namespace nn {

Linear::Linear(std::size_t input_size, std::size_t output_size, bool bias /* = true */)
    : num_in(input_size), num_out(output_size), bias(bias) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  auto weights = Variable<double>(Matrix<double>(num_in, num_out));
  std::generate(weights.value().begin(), weights.value().end(), [&dis, &gen] { return dis(gen); });

  if (bias) {
    auto bias_weights = Variable<double>(Matrix<double>(1, num_out));
    std::generate(bias_weights.value().begin(), bias_weights.value().end(),
                  [&dis, &gen] { return dis(gen); });
    _params = {weights, bias_weights};
  } else {
    _params = {weights};
  }
}

std::vector<Variable<double>> Linear::forward(const std::vector<Variable<double>>& inputs) {
  assert(inputs.size() == 1);

  auto weights = _params[0];
  auto bias = _params[1];

  auto x = ag::matmul(inputs[0], weights);
  x = x + bias;

  return {x};
}

}  // namespace nn
