add_library(nn
  nn/module.hpp
  nn/sequential.hpp
  nn/sequential.cpp
  nn/modules/linear.hpp
  nn/modules/linear.cpp
  nn/modules/sigmoid.hpp
  nn/modules/sigmoid.cpp
  nn/modules/relu.hpp
  nn/modules/relu.cpp
  nn/optim/sgd.hpp
  nn/mse.hpp
  nn/dataset/dataset.hpp
  nn/dataset/mnist_dataset.hpp
  nn/dataset/mnist_dataset.cpp

  matrix/matrix.hpp
  matrix/ops.hpp

  autograd/autograd.hpp
  autograd/functions.hpp
  autograd/variable.hpp
)

target_include_directories(nn
  PUBLIC .
)

target_compile_options(nn
  PRIVATE
    -Wall -Wextra -Wpedantic -std=c++20 -g -Og -fsanitize=address -fsanitize=undefined
)
target_link_options(nn
  PRIVATE
    -fsanitize=address -fsanitize=undefined
)
