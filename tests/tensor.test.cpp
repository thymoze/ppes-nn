#include <algorithm>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <random>
#include <tensor/tensor.hpp>
#include <vector>

TEST_CASE("benchmark") {
  auto t1 = Tensor<double>::rand({50, 50}, -10, 10);
  auto t2 = Tensor<double>::rand({50, 50}, -10, 10);

  t1.to(TensorBackend<double>(MTOps<double>()));
  t2.to(t1.f());
  BENCHMARK("MTOps MatMul") { return matmul(t1, t2); };

  t1.to(TensorBackend<double>(SimpleOps<double>()));
  t2.to(t1.f());
  BENCHMARK("SimpleOps MatMul") { return matmul(t1, t2); };
}

TEST_CASE("matmul") {
  auto backend = TensorBackend<double>(MTOps<double>());
  auto t1 = Tensor<double>::make({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto t2 = Tensor<double>::make({3, 2}, {2, 4, 6, 8, 10, 12});

  auto result = matmul(t1, t2);
  std::cout << result << std::endl;

  std::vector<double> expected{44, 56, 98, 128, 152, 200};
  REQUIRE(std::equal(result.data().data()->begin(), result.data().data()->end(), expected.begin(),
                     expected.end()));
}
