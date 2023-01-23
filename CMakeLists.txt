cmake_minimum_required(VERSION 3.15)

project(ppes_nn)
include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_subdirectory(nn)
add_subdirectory(src)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
