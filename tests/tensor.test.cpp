#include <algorithm>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <random>
#include <tensor/tensor.hpp>
#include <vector>

// TEST_CASE("benchmark") {
//   auto t1 = Tensor<double>::rand({50, 50}, -10, 10);
//   auto t2 = Tensor<double>::rand({50, 50}, -10, 10);
//
//   t1.to(TensorBackend<double>(MTOps<double>()));
//   t2.to(t1.f());
//   BENCHMARK("MTOps MatMul") { return matmul(t1, t2); };
//
//   t1.to(TensorBackend<double>(SimpleOps<double>()));
//   t2.to(t1.f());
//   BENCHMARK("SimpleOps MatMul") { return matmul(t1, t2); };
// }

// TEST_CASE("matmul") {
//   auto backend = TensorBackend<double>(MTOps<double>());
//   auto t1 = Tensor<double>::make({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
//   t1.requires_grad(true);
//   auto t2 = Tensor<double>::make({3, 2}, {4, 5, 6, 7, 8, 9});
//   t2.requires_grad(true);

//   auto result = matmul(t1, t2);
//   std::cout << result << std::endl;
//   mean(result).backward();
//   std::cout << *t1.grad() << std::endl;
//   std::cout << *t2.grad() << std::endl;

//   std::vector<double> expected{44, 56, 98, 128, 152, 200};
//   REQUIRE(std::equal(result.data().data()->begin(), result.data().data()->end(),
//   expected.begin(),
//                      expected.end()));
// }

// TEST_CASE("sigmoid") {
//   auto t1 = Tensor<double>::make({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
//   t1.requires_grad(true);

//   auto result = sigmoid(t1);
//   std::cout << result << std::endl;
//   result.backward();
//   std::cout << *t1.grad() << std::endl;

//   std::vector<double> expected{44, 56, 98, 128, 152, 200};
//   REQUIRE(std::equal(result.data().data()->begin(), result.data().data()->end(),
//   expected.begin(),
//                      expected.end()));
// }

// TEST_CASE("argmax") {
//   auto t1 = Tensor<double>::make({3, 3}, {3, 2, 1, 4, 5, 6, 7, 9, 8});

//   auto result = argmax(t1, 0);
//   std::cout << result << std::endl;

//   std::vector<double> expected{44, 56, 98, 128, 152, 200};
//   REQUIRE(std::equal(result.data().data()->begin(), result.data().data()->end(),
//   expected.begin(),
//                      expected.end()));
// }
