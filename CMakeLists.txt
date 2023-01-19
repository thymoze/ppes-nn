cmake_minimum_required(VERSION 3.15)

project(ppes_nn)

add_executable(main 
    src/main.cpp
    src/autograd/variable.cpp
    src/autograd/functions.cpp)
