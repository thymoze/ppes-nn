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

Linear::Linear(Variable<double> weights, Variable<double> bias_weights)
    : num_in(weights.rows()), num_out(weights.cols()), bias(true) {
  _params = {weights, bias_weights};
}

Linear::Linear(Variable<double> weights)
    : num_in(weights.rows()), num_out(weights.cols()), bias(false) {
  _params = {weights};
}

std::vector<Variable<double>> Linear::forward(const std::vector<Variable<double>>& inputs) {
  assert(inputs.size() == 1);

  auto weights = _params[0];
  auto bias = _params[1];

  auto x = ag::matmul(inputs[0], weights);
  x = x + bias;

  return {x};
}

std::string Linear::save(const std::string& model_name) {
  auto weights = _params[0];
  auto bias_weights = _params[1];
  std::string code = std::string("nn::Linear(Matrix<double>{") + std::to_string(weights.rows()) +
                     std::string(", ") + std::to_string(weights.cols()) + std::string(", {");
  for (auto val : weights.value()) {
    code += std::to_string(val) + ", ";
  }
  code.pop_back();
  code.pop_back();

  code += "}}, Matrix<double>{" + std::to_string(bias_weights.rows()) + std::string(", ") +
          std::to_string(bias_weights.cols()) + std::string(", {");

  for (auto val : bias_weights.value()) {
    code += std::to_string(val) + ", ";
  }
  code.pop_back();
  code.pop_back();

  code += "}})";
  return code;
}

}  // namespace nn
