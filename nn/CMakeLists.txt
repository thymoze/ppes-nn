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

    matrix/matrix.hpp
    matrix/ops.hpp

    autograd/autograd.hpp
    autograd/functions.hpp
    autograd/variable.hpp
)

target_include_directories(nn
    PUBLIC .
)