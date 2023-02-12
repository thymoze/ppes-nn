#include <iostream>
#include <nn/all.hpp>
#include <nn/quantization/quantization.hpp>
#include <string>
#include <tensor/quantization/quantized_tensor.hpp>
#include <utility>

#include "../trained_models/xor.hpp"
#include "../trained_models/xor_quant.hpp"

using tensor::Tensor;

int main(void) {
  auto model = nn::Sequential<float>();
  model.add(nn::Linear<float>(2, 3));
  model.add(nn::Sigmoid<float>());
  model.add(nn::Linear<float>(3, 1));
  model.init(xor_model::xor_model, xor_model::xor_model_len);

  for (auto& p : model.params()) {
    std::cout << p.template value<Tensor<float>>() << std::endl;
  }

  std::cout << std::endl;

  nn::quantization::quantize_dynamic(model);
  model.init(xor_quant_model::xor_quant_model, xor_quant_model::xor_quant_model_len);

  int i = 0;
  for (auto& p : model.params()) {
    if (i == 0 || i == 2) {
      std::cout << p.template value<tensor::quantization::QTensor>() << std::endl;
    } else {
      std::cout << p.template value<Tensor<float>>() << std::endl;
    }
    ++i;
  }
}
