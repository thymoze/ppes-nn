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
  set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL true)
  list(APPEND test_targets ${target})
  set(test_targets ${test_targets} PARENT_SCOPE)
endfunction()

################
# Add tests here

test_file(tensor_init_test tensor/tensor_init.test)
test_file(tensor_data_test tensor/tensor_data.test)
test_file(tensor_math_test tensor/tensor_math.test)
test_file(autodiff_test tensor/autodiff.test)
test_file(quantized_tensor_test tensor/quantization/quantized_tensor.test)
test_file(mse_test nn/loss/mse.test)
test_file(linear_test nn/modules/linear.test)
test_file(relu_test nn/modules/relu.test)
test_file(sequential_test nn/sequential.test)
test_file(sgd_test nn/optim/sgd.test)

################

add_custom_target(tests DEPENDS ${test_targets})
