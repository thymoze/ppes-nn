#include <catch2/catch_test_macros.hpp>
#include <tensor/quantization/quantized_tensor.hpp>

TEST_CASE("Quantization and dequantization") {
  auto x = tensor::rand<float>({2, 3, 4}, -5, 5);

  auto qparams = tensor::quantization::calc_quantization_params(x);
  auto q = tensor::quantization::quantize(x, qparams.scale, qparams.zero_point);

  auto dx = tensor::quantization::dequantize<float>(q);

  INFO("Result:\n" << x);
  INFO("Expected:\n" << dx);

  // Relatively large tolerance to account for quantization error
  REQUIRE(tensor::all(tensor::is_close(x, dx, 1e-1, 1e-1)).item());
}

TEST_CASE("Quantized matrix multiplication") {
  auto x = tensor::rand<float>({4, 1, 2, 2}, -5, 5);
  auto y = tensor::rand<float>({4, 3, 2, 5}, -5, 5);

  auto qxparams = tensor::quantization::calc_quantization_params(x);
  auto qx = tensor::quantization::quantize(x, qxparams.scale, qxparams.zero_point);
  auto dx = tensor::quantization::dequantize<float>(qx);

  auto qyparams = tensor::quantization::calc_quantization_params(y);
  auto qy = tensor::quantization::quantize(y, qyparams.scale, qyparams.zero_point);
  auto dy = tensor::quantization::dequantize<float>(qy);

  auto dres = tensor::matmul<float>(qx, qy);

  auto dexp = tensor::matmul(dx, dy);

  INFO("Result:\n" << dres);
  INFO("Expected:\n" << dexp);

  REQUIRE(tensor::all(tensor::is_close(dres, dexp)).item());
}
