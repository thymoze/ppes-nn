#pragma once

#include <nn/quantization/modules/quantized_linear.hpp>
#include <nn/sequential.hpp>
#include <tensor/quantization/quantized_tensor.hpp>

namespace nn {
namespace quantization {

template <typename T>
void quantize_dynamic(Sequential<T>& model) {
  for (auto& module : model.modules()) {
    auto linear = dynamic_cast<Linear<T>*>(module.get());
    if (linear != nullptr) {
      auto quant = QLinear<T>(linear->num_in(), linear->num_out(), linear->bias());

      auto& param = linear->params()[0];
      auto v = param.template value<Tensor<T>>();
      auto qp = tensor::quantization::calc_quantization_params(v);
      auto q = tensor::quantization::quantize(v, qp.scale, qp.zero_point);
      quant.params()[0].update(q);

      quant.params()[1] = linear->params()[1];
      module = std::make_shared<QLinear<T>>(std::move(quant));
    }
  }
}

}  // namespace quantization
}  // namespace nn
