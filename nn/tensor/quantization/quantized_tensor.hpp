#pragma once

#include <tensor/tensor.hpp>

namespace tensor {
namespace quantization {

struct QuantizationParams {
  float scale;
  std::uint8_t zero_point;
};

class QTensor : public Tensor<std::uint8_t> {
 public:
  QTensor(std::unique_ptr<TensorData<std::uint8_t>> storage, float scale, std::uint8_t zero_point)
      : Tensor<std::uint8_t>(std::move(storage)), qparams_{scale, zero_point} {}

  QTensor(Tensor<std::uint8_t> base, float scale, std::uint8_t zero_point)
      : Tensor<std::uint8_t>(std::move(base)), qparams_{scale, zero_point} {}

  [[nodiscard]] float scale() const { return qparams_.scale; }
  [[nodiscard]] std::uint8_t zero_point() const { return qparams_.zero_point; }

  [[nodiscard]] std::string to_string() const;

 private:
  QuantizationParams qparams_;
};

}  // namespace quantization
}  // namespace tensor

#include <tensor/quantization/quantization.hpp>
#include <tensor/quantization/quantized_ops.hpp>

namespace tensor {

using quantization::QTensor;

[[nodiscard]] std::string QTensor::to_string() const {
  std::stringstream out;
  out << "(QTensor <scale=" << qparams_.scale << ", zero_point=" << qparams_.zero_point << ">\n";
  out << tensor::quantization::dequantize<float>(*this).to_string() << ")";
  return out.str();
}

// Performs matrix-multiplication on quantized tensors, returning a de-quantized floating point
// result.
template <typename T>
Tensor<T> matmul(const QTensor& a, const QTensor& b) {
  assert(*(a.shape().end() - 1) == *(b.shape().end() - 2) &&
         "Matrix multiplication dimensions don't match.");

  QTensor lhs = a.ndims() == 2 ? QTensor(a.unsqueeze(0), a.scale(), a.zero_point()) : a;
  QTensor rhs = b.ndims() == 2 ? QTensor(b.unsqueeze(0), b.scale(), b.zero_point()) : b;
  bool both_2d = a.ndims() == 2 && b.ndims() == 2;

  auto shape = broadcast_shapes_for_matmul(lhs.shape(), rhs.shape());

  auto out = tensor::zeros<T>(shape);
  quantization::matrix_multiply<T>(lhs, rhs, out);

  if (both_2d) {
    out = out.view({*(shape.end() - 2), *(shape.end() - 1)});
  }

  return out;
}

}  // namespace tensor
