#pragma once

#include <algorithm>
#include <nn/module.hpp>

namespace nn {

using namespace tensor;

template <typename T, std::size_t In, std::size_t Out>
class Linear : public Module<T> {
 public:
  Linear() : Linear(true) {}
  Linear(bool bias) : bias_(bias) {
    this->params_ = {Parameter<T>()};

    if (bias) {
      this->params_.emplace_back();
    }
  }

  void init() override {
    auto bound = 1 / std::sqrt(In);
    this->params_[0].update(tensor::rand<T>({In, Out}, -bound, bound));
    if (bias_) {
      this->params_[1].update(tensor::rand<T>({Out}, -bound, bound));
    }
  }

  unsigned int init(const unsigned char data[], const unsigned int data_len) override {
    unsigned int size = (In * Out + (bias_ ? Out : 0)) * sizeof(T);
    if (size > data_len) {
      throw std::logic_error("Cannot initialize module of size " + std::to_string(size) +
                             " with array of length " + std::to_string(data_len));
    }

    using WDataPtr = const T(*)[In * Out];
    WDataPtr w_arr = reinterpret_cast<WDataPtr>(data);
    auto w_storage = TensorStorage<T, WDataPtr>(w_arr, {In, Out});
    this->params_[0].update(Tensor<T>(std::move(w_storage)));

    if (bias_) {
      using BDataPtr = const T(*)[Out];
      BDataPtr b_arr = reinterpret_cast<BDataPtr>(data + sizeof(*w_arr));
      auto b_storage = TensorStorage<T, BDataPtr>(b_arr, {Out});
      this->params_[1].update(Tensor<T>(std::move(b_storage)));
    }

    return size;
  }

  Tensor<T> forward(const Tensor<T>& input) override {
    auto weights = this->params_[0];
    auto bias = this->params_[1];

    auto x = matmul(input, weights.value());
    x = x + bias.value();

    return x;
  }

 private:
  std::size_t num_in_;
  std::size_t num_out_;
  bool bias_;
};

}  // namespace nn
