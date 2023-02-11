#pragma once

#include <algorithm>
#include <autograd/autograd.hpp>
#include <matrix/matrix.hpp>
#include <nn/module.hpp>
#include <random>

namespace nn {

template <typename T>
class Linear : public Module<T> {
 public:
  Linear(std::size_t input_size, std::size_t output_size, bool bias = true)
      : num_in_(input_size), num_out_(output_size), bias_(bias) {
    std::random_device rd;
    std::mt19937 gen(rd());
    auto bound = 1 / std::sqrt(input_size);
    std::uniform_real_distribution<> dis(-bound, bound);

    auto weights = Variable<T>(Matrix<T>(num_in_, num_out_));
    std::generate(weights.value().begin(), weights.value().end(),
                  [&dis, &gen] { return dis(gen); });

    if (bias) {
      auto bias_weights = Variable<T>(Matrix<T>(1, num_out_));
      std::generate(bias_weights.value().begin(), bias_weights.value().end(),
                    [&dis, &gen] { return dis(gen); });
      this->params_ = {weights, bias_weights};
    } else {
      this->params_ = {weights};
    }
  }

  Linear(Variable<T> weights, Variable<T> bias_weights)
      : num_in_(weights.rows()), num_out_(weights.cols()), bias_(true) {
    this->params_ = {weights, bias_weights};
  }

  Linear(Variable<T> weights) : num_in_(weights.rows()), num_out_(weights.cols()), bias_(false) {
    this->params_ = {weights};
  }

  std::vector<Variable<T>> forward(const std::vector<Variable<T>>& inputs) override {
    assert(inputs.size() == 1);

    auto weights = this->params_[0];
    auto bias = this->params_[1];

    auto x = matmul(inputs[0], weights);
    x = x + bias;

    return {x};
  }

  std::string save(const std::string&) override {
    auto weights = this->params_[0];
    auto bias_weights = this->params_[1];
    std::string code = std::string("nn::Linear<T>(nn::Matrix<T>{") +
                       std::to_string(weights.rows()) + std::string(", ") +
                       std::to_string(weights.cols()) + std::string(", {");
    for (auto val : weights.value()) {
      code += std::to_string(val) + ", ";
    }
    code.pop_back();
    code.pop_back();

    code += "}}, nn::Matrix<T>{" + std::to_string(bias_weights.rows()) + std::string(", ") +
            std::to_string(bias_weights.cols()) + std::string(", {");

    for (auto val : bias_weights.value()) {
      code += std::to_string(val) + ", ";
    }
    code.pop_back();
    code.pop_back();

    code += "}})";
    return code;
  };

  bool is_prunable() override { return true; }

  void prune_one_neuron() override {
    std::cout << "hello in linear" << std::endl;
    nn::Variable<T> weigths = this->params_[0];
    int test = weigths.value().lowest_row_sum();
    weigths.value().delete_row(test);
  }

 private:
  Linear() = default;

  std::size_t num_in_;
  std::size_t num_out_;
  bool bias_;
};

}  // namespace nn
