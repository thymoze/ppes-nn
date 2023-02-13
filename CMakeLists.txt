cmake_minimum_required(VERSION 3.15)

include(cmake/link_binary.cmake)

if(BUILD_PICO)
  set(BUILD_TESTING false)
  include(cmake/pico_sdk.cmake)
endif()

project(ppes_nn C CXX ASM)
include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_subdirectory(nn)
add_subdirectory(src)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
