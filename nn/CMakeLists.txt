add_library(nn
  nn/module.hpp
  nn/sequential.hpp
  nn/modules/linear.hpp
  nn/modules/sigmoid.hpp
  nn/modules/softmax.hpp
  nn/modules/relu.hpp
  nn/optim/sgd.hpp
  nn/loss/mse.hpp
  nn/dataset/dataset.hpp
  nn/dataset/mnist_dataset.hpp
  nn/quantization/quantization.hpp
  nn/quantization/modules/quantized_linear.hpp
  nn/all.hpp

  tensor/tensor.hpp
  tensor/tensor_data.hpp
  tensor/tensor_ops.hpp
  tensor/tensor_functions.hpp
  tensor/tensor_util.hpp
  tensor/autodiff.hpp
  tensor/quantization/quantization.hpp
  tensor/quantization/quantized_ops.hpp
  tensor/quantization/quantized_tensor.hpp
)

target_include_directories(nn
  PUBLIC .
)

set_target_properties(nn PROPERTIES LINKER_LANGUAGE CXX)
