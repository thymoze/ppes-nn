#pragma once

#include <algorithm>
#include <autograd/autograd.hpp>
#include <matrix/matrix.hpp>
#include <nn/module.hpp>
#include <random>
#include <sstream>

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
    std::stringstream code_stream;
    auto weights = this->params_[0];
    code_stream << "nn::Linear<T>(nn::Matrix<T>{" << weights.rows() << ", " << weights.cols()
                << ", {";
    for (auto val : weights.value()) {
      code_stream << val << ", ";
    }

    code_stream.seekp(-2, code_stream.cur);

    if (bias_) {
      auto bias_weights = this->params_[1];
      code_stream << "}}, nn::Matrix<T>{" << bias_weights.rows() << ", " << bias_weights.cols()
                  << ", {";
      for (auto val : bias_weights.value()) {
        code_stream << val << ", ";
      }
      code_stream.seekp(-2, code_stream.cur);
    }
    code_stream << "}})";
    return code_stream.str();
  };

  bool is_prunable() override { return this->params_[0].value().rows() > 1; }

  int prune_one_neuron() override {
    nn::Variable<T> weigths = this->params_[0];
    int test = weigths.value().lowest_row_sum_index();
    weigths.value().delete_row(test);
    return test;
  }

  void apply_pruned_neuron(int neuron) override {
    auto weights = this->params_[0];
    std::cout << "column to delete" << neuron << std::endl;
    weights.value().delete_column(neuron);
    if (bias_) {
      auto bias_weights = this->params_[1];
      bias_weights.value().delete_column(neuron);
    }
  }
  bool is_linear() override { return true; }

 private:
  Linear() = default;

  std::size_t num_in_;
  std::size_t num_out_;
  bool bias_;
};

}  // namespace nn
