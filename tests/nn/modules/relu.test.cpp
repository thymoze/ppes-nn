#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <nn/all.hpp>
#include <tensor/tensor.hpp>

TEST_CASE("forward") {
  auto relu = nn::ReLU<float>();
  auto [src, expected] = GENERATE(std::pair{tensor::make<float>({2, 2}, {1, 2, 3, 4}),
                                            tensor::make<float>({2, 2}, {1, 2, 3, 4})},
                                  std::pair{tensor::make<float>({2, 2}, {-1, -2, -3, -4}),
                                            tensor::make<float>({2, 2}, {0, 0, 0, 0})},
                                  std::pair{tensor::make<float>({2, 2}, {1, -2, 3, -4}),
                                            tensor::make<float>({2, 2}, {1, 0, 3, 0})},
                                  std::pair{tensor::make<float>({2, 2}, {1, 0, -3, 4}),
                                            tensor::make<float>({2, 2}, {1, 0, 0, 4})});

  auto actual = relu.forward(src);

  REQUIRE(std::equal(actual.data()->begin(), actual.data()->end(), expected.data()->begin()));
  REQUIRE(std::equal(actual.shape().begin(), actual.shape().end(), expected.shape().begin()));
}
