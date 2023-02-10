include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.3.0
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(tensor_test tensor.test.cpp)
target_link_libraries(tensor_test PRIVATE Catch2::Catch2WithMain nn)
catch_discover_tests(tensor_test)

add_executable(quantized_tensor_test tensor/quantization/quantized_tensor.test.cpp)
target_link_libraries(quantized_tensor_test PRIVATE Catch2::Catch2WithMain nn)
catch_discover_tests(quantized_tensor_test)
