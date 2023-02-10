#pragma once

#include <tensor/quantization/quantized_tensor.hpp>
#include <tensor/tensor_ops.hpp>

namespace tensor {
namespace quantization {

template <typename T, typename R, typename S>
Tensor<R> reduce(const Tensor<T>& input, std::function<S(T, T)> fn, S start, std::size_t dim) {
  auto shape = input.shape();
  shape[dim] = 1;
  auto out = tensor::zeros<R>(shape);

  auto dim_size = shape[dim];
  for (auto& idx : out.indices()) {
    auto in_idx = idx;

    S v = start;
    for (std::size_t i = 0; i < dim_size; i++) {
      in_idx[dim] = i;
      v = fn(v, input[in_idx]);
    }

    out[idx] = v;
  }
  return out;
}

template <typename T>
void matrix_multiply(const QTensor& lhs, const QTensor& rhs, Tensor<T>& out) {
  Indices lhs_idx, rhs_idx;
  for (auto& idx : out.indices()) {
    std::uint32_t v = 0;
    for (std::size_t k = 0; k < lhs.shape().back(); k++) {
      broadcasted_to_index_in_shape(idx, lhs.shape(), lhs_idx);
      *(lhs_idx.end() - 1) = k;

      broadcasted_to_index_in_shape(idx, rhs.shape(), rhs_idx);
      *(rhs_idx.end() - 2) = k;

      v += lhs[lhs_idx] * rhs[rhs_idx];
    }
    out[idx] = v;
  }
  out = out + (reduce<std::uint8_t, T, std::uint32_t>(rhs, std::plus(), 0, rhs.ndims() - 2) *
               tensor::make<T>(-rhs.zero_point()));
  out = out + (reduce<std::uint8_t, T, std::uint32_t>(lhs, std::plus(), 0, lhs.ndims() - 1) *
               tensor::make<T>(-lhs.zero_point()));
  out = out + tensor::make<T>(lhs.shape().back());
  out = out * tensor::make<T>(lhs.scale() * rhs.scale());
}

}  // namespace quantization
}  // namespace tensor
