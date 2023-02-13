#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <tensor/tensor.hpp>

using Catch::Matchers::AllMatch;
using Catch::Matchers::RangeEquals;
using Catch::Matchers::WithinAbs;

TEST_CASE("Init zeros") {
  auto t = tensor::zeros<int>({2, 3});

  CHECK_THAT(t.shape(), RangeEquals(std::vector<int>{2, 3}));
  CHECK_THAT(*t.data(), AllMatch(WithinAbs(0, 0)));
}

TEST_CASE("Init ones") {
  auto t = tensor::ones<int>({5, 1, 9});

  CHECK_THAT(t.shape(), RangeEquals(std::vector<int>{5, 1, 9}));
  CHECK_THAT(*t.data(), AllMatch(WithinAbs(1, 0)));
}

TEST_CASE("Init eye") {
  auto t = tensor::eye<int>(3);

  CHECK_THAT(t.shape(), RangeEquals(std::vector<int>{3, 3}));
  CHECK_THAT(*t.data(), RangeEquals(std::vector<int>{1, 0, 0, 0, 1, 0, 0, 0, 1}));
}

TEST_CASE("Init rand") {
  auto t = tensor::rand<int>({4, 1}, -1, 1);

  CHECK_THAT(t.shape(), RangeEquals(std::vector<int>{4, 1}));
  CHECK_THAT(*t.data(), AllMatch(WithinAbs(0, 1)));
}

TEST_CASE("Make") {
  auto t = tensor::make<int>(25);
  CHECK_THAT(t.shape(), RangeEquals(std::vector<int>{1}));
  CHECK_THAT(*t.data(), RangeEquals(std::vector<int>{25}));

  auto t2 = tensor::make<int>({10, 9, 8, 7, 6, 5});
  CHECK_THAT(t2.shape(), RangeEquals(std::vector<int>{6}));
  CHECK_THAT(*t2.data(), RangeEquals(std::vector<int>{10, 9, 8, 7, 6, 5}));

  auto t3 = tensor::make<int>({3, 2, 1}, {1, 2, 3, 4, 5, 6});
  CHECK_THAT(t3.shape(), RangeEquals(std::vector<int>{3, 2, 1}));
  CHECK_THAT(*t3.data(), RangeEquals(std::vector<int>{1, 2, 3, 4, 5, 6}));
}
