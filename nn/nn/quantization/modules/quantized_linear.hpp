#pragma once

#include <nn/modules/linear.hpp>
#include <tensor/quantization/quantized_tensor.hpp>

namespace nn {
namespace quantization {

template <typename T>
class QLinear : public Linear<T> {
 public:
  QLinear(std::size_t num_in, std::size_t num_out, bool bias = true)
      : Linear<T>(num_in, num_out, bias) {}

  void init() override {
    throw std::logic_error("Quantized modules can only be initialized from existing data.");
  }

  unsigned int init(const unsigned char data[], const unsigned int data_len) override {
    unsigned int offset = 0;

    float scale = *reinterpret_cast<const float*>(data);
    offset += sizeof(scale);

    std::uint8_t zero_point = *(data + offset);
    offset += 1;

    const std::uint8_t* w_arr = reinterpret_cast<const std::uint8_t*>(data + offset);
    offset += this->num_in_ * this->num_out_ * sizeof(std::uint8_t);
    auto w_storage =
        std::make_unique<PointerStorage<std::uint8_t>>(w_arr, Shape{this->num_in_, this->num_out_});
    this->params_[0].update(QTensor(std::move(w_storage), scale, zero_point));

    // skip some bytes, so the next read of T will be correctly aligned
    auto alignment = std::alignment_of_v<T>;
    offset += alignment - (offset % alignment);

    if (this->bias_) {
      const T* b_arr = reinterpret_cast<const T*>(data + offset);
      offset += this->num_out_ * sizeof(T);
      auto b_storage = std::make_unique<PointerStorage<T>>(b_arr, Shape{1, this->num_out_});
      this->params_[1].update(Tensor<T>(std::move(b_storage)));
    }

    if (offset > data_len) {
      throw std::logic_error("Cannot initialize module of size " + std::to_string(offset) +
                             " with array of length " + std::to_string(data_len));
    }

    return offset;
  }

  std::vector<std::uint8_t> data() override {
    std::vector<std::uint8_t> result;

    auto& t = this->params_[0].template value<QTensor>();
    auto scale = t.scale();
    auto* p_scale = reinterpret_cast<std::uint8_t*>(&scale);
    for (std::size_t s = 0; s < sizeof(scale); ++s) {
      result.push_back(*(p_scale + s));
    }

    result.push_back(t.zero_point());

    auto& data = *t.data();
    for (auto& v : data) {
      auto* p = reinterpret_cast<const std::uint8_t*>(&v);
      for (std::size_t s = 0; s < sizeof(v); ++s) {
        result.push_back(*(p + s));
      }
    }
    // make sure the next value will be correctly aligned
    auto alignment = std::alignment_of_v<T>;
    auto missing = alignment - (result.size() % alignment);
    result.insert(result.end(), missing, 0);

    if (this->bias_) {
      auto& data = *this->params_[1].template value<Tensor<T>>().data();
      for (auto& v : data) {
        auto* p = reinterpret_cast<const std::uint8_t*>(&v);
        for (std::size_t s = 0; s < sizeof(v); ++s) {
          result.push_back(*(p + s));
        }
      }
    }

    return result;
  }

  Tensor<T> forward(const Tensor<T>& input) override {
    auto weights = this->params_[0];
    auto bias = this->params_[1];

    auto inparams = tensor::quantization::calc_quantization_params(input);
    auto qin = tensor::quantization::quantize(input, inparams.scale, inparams.zero_point);
    auto x = matmul<T>(qin, weights.template value<QTensor>());
    x = x + bias.template value<Tensor<T>>();

    return x;
  }
};

}  // namespace quantization
}  // namespace nn
