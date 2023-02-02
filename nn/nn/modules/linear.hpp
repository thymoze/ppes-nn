#pragma once

#include <algorithm>
#include <nn/module.hpp>

namespace nn {

template <typename T>
class Linear : public Module<T> {
 public:
  Linear(std::size_t input_size, std::size_t output_size, bool bias = true)
      : num_in_(input_size), num_out_(output_size), bias_(bias) {
    auto bound = 1 / std::sqrt(input_size);

    auto weights = Tensor<T>::rand({num_in_, num_out_}, -bound, bound);

    if (bias) {
      auto bias_weights = Tensor<T>::rand({num_out_}, -bound, bound);
      this->params_ = {Parameter<T>(std::move(weights)), Parameter<T>(std::move(bias_weights))};
    } else {
      this->params_ = {Parameter<T>(std::move(weights))};
    }
  }

  // Linear(Tensor<T> weights, Tensor<T> bias_weights)
  //     : num_in_(weights.rows()), num_out_(weights.cols()), bias_(true) {
  //   this->params_ = {Parameter<T>(std::move(weights)), Parameter<T>(std::move(bias_weights))};
  // }

  // Linear(Tensor<T> weights) : num_in_(weights.rows()), num_out_(weights.cols()), bias_(false) {
  //   this->params_ = {Parameter<T>(std::move(weights))};
  // }

  Tensor<T> forward(const Tensor<T>& input) override {
    auto weights = this->params_[0];
    auto bias = this->params_[1];

    auto x = matmul(input, weights.value());
    x = x + bias.value();

    return x;
  }

  std::string save(const std::string&) override{
      // auto weights = this->params_[0];
      // auto bias_weights = this->params_[1];
      // std::string code = std::string("nn::Linear<T>(nn::Matrix<T>{") +
      //                    std::to_string(weights.rows()) + std::string(", ") +
      //                    std::to_string(weights.cols()) + std::string(", {");
      // for (auto val : weights.value()) {
      //   code += std::to_string(val) + ", ";
      // }
      // code.pop_back();
      // code.pop_back();

      // code += "}}, nn::Matrix<T>{" + std::to_string(bias_weights.rows()) + std::string(", ") +
      //         std::to_string(bias_weights.cols()) + std::string(", {");

      // for (auto val : bias_weights.value()) {
      //   code += std::to_string(val) + ", ";
      // }
      // code.pop_back();
      // code.pop_back();

      // code += "}})";
      // return code;
  };

 private:
  Linear() = default;

  std::size_t num_in_;
  std::size_t num_out_;
  bool bias_;
};

}  // namespace nn
