cmake_minimum_required(VERSION 3.15)

project(ppes_nn)

add_executable(main 
    src/main.cpp
    src/nn/sequential.cpp
    src/nn/modules/linear.cpp
    src/nn/modules/relu.cpp
    src/nn/modules/sigmoid.cpp)

set_target_properties(main
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED true
)

target_include_directories(main
    PRIVATE
        src
)
