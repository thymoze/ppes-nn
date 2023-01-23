cmake_minimum_required(VERSION 3.15)

project(ppes_nn)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_subdirectory(nn)
add_subdirectory(src)
add_subdirectory(tests)