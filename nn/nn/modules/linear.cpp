#include "linear.hpp"

#include <algorithm>
#include <autograd/autograd.hpp>
#include <matrix/matrix.hpp>
#include <random>

namespace nn {

Linear::Linear(std::size_t input_size, std::size_t output_size, bool bias /* = true */)
    : num_in_(input_size), num_out_(output_size), bias_(bias) {
  std::random_device rd;
  std::mt19937 gen(rd());
  auto bound = 1 / std::sqrt(input_size);
  std::uniform_real_distribution<> dis(-bound, bound);

  auto weights = Variable<double>(Matrix<double>(num_in_, num_out_));
  std::generate(weights.value().begin(), weights.value().end(), [&dis, &gen] { return dis(gen); });

  if (bias) {
    auto bias_weights = Variable<double>(Matrix<double>(1, num_out_));
    std::generate(bias_weights.value().begin(), bias_weights.value().end(),
                  [&dis, &gen] { return dis(gen); });
    params_ = {weights, bias_weights};
  } else {
    params_ = {weights};
  }
}

Linear::Linear(Variable<double> weights, Variable<double> bias_weights)
    : num_in_(weights.rows()), num_out_(weights.cols()), bias_(true) {
  params_ = {weights, bias_weights};
}

Linear::Linear(Variable<double> weights)
    : num_in_(weights.rows()), num_out_(weights.cols()), bias_(false) {
  params_ = {weights};
}

std::vector<Variable<double>> Linear::forward(const std::vector<Variable<double>>& inputs) {
  assert(inputs.size() == 1);

  auto weights = params_[0];
  auto bias = params_[1];

  auto x = matmul(inputs[0], weights);
  x = x + bias;

  return {x};
}

std::string Linear::save(const std::string& model_name) {
  auto weights = params_[0];
  auto bias_weights = params_[1];
  std::string code = std::string("nn::Linear(nn::Matrix<double>{") +
                     std::to_string(weights.rows()) + std::string(", ") +
                     std::to_string(weights.cols()) + std::string(", {");
  for (auto val : weights.value()) {
    code += std::to_string(val) + ", ";
  }
  code.pop_back();
  code.pop_back();

  code += "}}, nn::Matrix<double>{" + std::to_string(bias_weights.rows()) + std::string(", ") +
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
