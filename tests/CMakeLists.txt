include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.3.0
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

function(test_file target file)
  add_executable(${target} ${file}.cpp)
  target_compile_options(${target} PRIVATE -g -fsanitize=address,undefined)
  target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  target_link_libraries(${target} PRIVATE Catch2::Catch2WithMain nn)
  catch_discover_tests(${target})
endfunction()

test_file(tensor_test tensor/tensor.test)
test_file(quantized_tensor_test tensor/quantization/quantized_tensor.test)

