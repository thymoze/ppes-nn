find_package(TBB REQUIRED)

add_library(nn
  nn/module.hpp
  nn/sequential.hpp
  nn/modules/linear.hpp
  nn/modules/sigmoid.hpp
  nn/modules/relu.hpp
  nn/optim/sgd.hpp
  nn/mse.hpp
  nn/dataset/dataset.hpp
  nn/dataset/mnist_dataset.hpp

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
    -Wall -Wextra -Wpedantic -std=c++17 -g -Og -fsanitize=address -fsanitize=undefined
)
target_link_options(nn
  PRIVATE
    -fsanitize=address -fsanitize=undefined
)

target_link_libraries(nn PRIVATE TBB::tbb)

set_target_properties(nn PROPERTIES LINKER_LANGUAGE CXX)
