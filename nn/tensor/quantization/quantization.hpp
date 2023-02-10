#pragma once

#include <tensor/quantization/quantized_tensor.hpp>

namespace tensor {
namespace quantization {

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
QuantizationParams calc_quantization_params(const Tensor<T>& input) {
  auto min = tensor::min(input).item();
  auto max = tensor::max(input).item();

  const float qmin = 0;
  const float qmax = 255;

  const double scale = (max - min) / (qmax - qmin);

  const double initial_zero_point = qmin - min / scale;

  std::uint8_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) {
    nudged_zero_point = qmin;
  } else if (initial_zero_point > qmax) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = static_cast<std::uint8_t>(std::round(initial_zero_point));
  }

  return QuantizationParams{static_cast<float>(scale), nudged_zero_point};
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
QTensor quantize(const Tensor<T>& input, float scale, std::uint8_t zero_point) {
  auto transformed = tensor::make<T>(zero_point) + input / tensor::make<T>(scale);
  auto data = std::make_shared<std::vector<std::uint8_t>>();
  data->reserve(transformed.size());
  std::transform(
      transformed.data()->begin(), transformed.data()->end(), std::back_inserter(*data),
      [](auto v) { return static_cast<std::uint8_t>(std::round(std::clamp(v, (T)0, (T)255))); });
  auto storage = std::make_unique<VectorStorage<std::uint8_t>>(
      std::move(data), transformed.strides(), transformed.shape());
  return QTensor(std::move(storage), scale, zero_point);
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
Tensor<T> dequantize(const QTensor& input) {
  auto data = std::make_shared<std::vector<T>>();
  data->reserve(input.size());
  auto s = input.scale();
  auto z = input.zero_point();
  std::transform(input.data()->begin(), input.data()->end(), std::back_inserter(*data),
                 [s, z](auto v) { return static_cast<T>(v - z) * s; });
  auto storage =
      std::make_unique<VectorStorage<T>>(std::move(data), input.strides(), input.shape());
  return Tensor<T>(std::move(storage));
}

}  // namespace quantization
}  // namespace tensor
