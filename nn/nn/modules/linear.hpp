#pragma once

#include <algorithm>
#include <memory>
#include <nn/module.hpp>
#include <random>

namespace nn {

using namespace tensor;

template <typename T>
class Linear : public Module<T> {
 public:
  Linear(std::size_t in, std::size_t out, bool bias = true)
      : num_in_(in), num_out_(out), bias_(bias) {
    this->params_.emplace_back();

    if (bias) {
      this->params_.emplace_back();
    }
  }

  void init() override {
    auto bound = 1 / std::sqrt(num_in_);
    this->params_[0].update(tensor::rand<T>({num_in_, num_out_}, -bound, bound));
    if (bias_) {
      this->params_[1].update(tensor::rand<T>({1, num_out_}, -bound, bound));
    }
  }

  unsigned int init(const unsigned char data[], const unsigned int data_len) override {
    unsigned int size = (num_in_ * num_out_ + (bias_ ? num_out_ : 0)) * sizeof(T);
    if (size > data_len) {
      throw std::logic_error("Cannot initialize module of size " + std::to_string(size) +
                             " with array of length " + std::to_string(data_len));
    }

    const T* w_arr = reinterpret_cast<const T*>(data);
    auto w_storage = std::make_unique<PointerStorage<T>>(w_arr, Shape{num_in_, num_out_});
    this->params_[0].update(Tensor<T>(std::move(w_storage)));

    if (bias_) {
      const T* b_arr = reinterpret_cast<const T*>(data + (num_in_ * num_out_ * sizeof(T)));
      auto b_storage = std::make_unique<PointerStorage<T>>(b_arr, Shape{num_out_});
      this->params_[1].update(Tensor<T>(std::move(b_storage)));
    }

    return size;
  }

  Tensor<T> forward(const Tensor<T>& input) override {
    auto weights = this->params_[0];
    auto bias = this->params_[1];

    auto x = matmul(input, weights.template value<Tensor<T>>());
    x = x + bias.template value<Tensor<T>>();

    return x;
  }

  bool is_prunable() override {
    return this->params_[0].template value<Tensor<T>>().shape()[0] > 1;
  }

  int prune_one_neuron() override {
    auto& weights = this->params_[0].template value<Tensor<T>>();

    auto abssums = tensor::min(tensor::abssum(weights, 0));
    std::size_t idx = tensor::argmin(abssums).item();

    this->params_[0].update(weights.remove(0, idx));

    num_in_ -= 1;
    return idx;
  }

  void apply_pruned_neuron(int neuron) override {
    auto& weights = this->params_[0].template value<Tensor<T>>();

    this->params_[0].update(weights.remove(1, neuron));
    if (bias_) {
      auto& bias_weights = this->params_[1].template value<Tensor<T>>();

      this->params_[1].update(bias_weights.remove(1, neuron));
    }
    num_out_ -= 1;
  }

  bool is_linear() override { return true; }

  std::size_t num_in() const { return num_in_; }
  std::size_t num_out() const { return num_out_; }
  bool bias() const { return bias_; }

 protected:
  std::size_t num_in_;
  std::size_t num_out_;
  bool bias_;

  Linear() = default;
};

}  // namespace nn
