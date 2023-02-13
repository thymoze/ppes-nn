#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <tensor/tensor.hpp>

using Catch::Matchers::RangeEquals;

TEST_CASE("Negation backward pass") {
  auto t = tensor::zeros<int>({2, 2});
  t.requires_grad(true);
  auto r = tensor::Neg<int>()(t);

  r.backward(tensor::make<int>({2, 2}, {2, -2, 4, -4}));

  REQUIRE(t.grad().has_value());
  CHECK_THAT((*t.grad()).shape(), RangeEquals(std::vector<int>{2, 2}));
  CHECK_THAT(*(*t.grad()).data(), RangeEquals(std::vector<int>{-2, 2, -4, 4}));
}

TEST_CASE("Inverse backward pass") {
  auto t = tensor::make<float>({4, 1}, {1, 2, 3, 4});
  t.requires_grad(true);
  auto r = tensor::Inv<float>()(t);

  r.backward(tensor::make<float>({4, 1}, {4, 3, 2, 1}));

  REQUIRE(t.grad().has_value());
  CHECK_THAT((*t.grad()).shape(), RangeEquals(std::vector<int>{4, 1}));
  CHECK_THAT(*(*t.grad()).data(), RangeEquals(std::vector<float>{-4, -3. / 4, -2. / 9, -1. / 16}));
}

TEST_CASE("ReLU backward pass") {
  auto t = tensor::make<float>({1, 4}, {1, -1, 0.1, -0.1});
  t.requires_grad(true);
  auto r = tensor::ReLU<float>()(t);

  r.backward(tensor::make<float>({1, 4}, {4, 3, 2, 1}));

  REQUIRE(t.grad().has_value());
  CHECK_THAT((*t.grad()).shape(), RangeEquals(std::vector<int>{1, 4}));
  CHECK_THAT(*(*t.grad()).data(), RangeEquals(std::vector<float>{4, 0, 2, 0}));
}

TEST_CASE("Exp backward pass") {
  auto t = tensor::make<float>({1, 2}, {0, 1});
  t.requires_grad(true);
  auto r = tensor::Exp<float>()(t);

  r.backward(tensor::make<float>({1, 2}, {2, 3}));

  REQUIRE(t.grad().has_value());
  CHECK_THAT((*t.grad()).shape(), RangeEquals(std::vector<int>{1, 2}));
  CHECK_THAT(*(*t.grad()).data(), RangeEquals(std::vector<float>{2, 3 * std::exp(1)}));
}

TEST_CASE("Sigmoid backward pass") {
  auto t = tensor::make<float>({-1, 0, 1});
  t.requires_grad(true);
  auto r = tensor::Sigmoid<float>()(t);

  r.backward(tensor::make<float>({2, 3, 4}));

  REQUIRE(t.grad().has_value());
  CHECK_THAT((*t.grad()).shape(), RangeEquals(std::vector<int>{3}));
  CHECK_THAT(*(*t.grad()).data(),
             RangeEquals(std::vector<float>{0.39322, 0.7500, 0.78645},
                         [](auto l, auto r) { return std::abs(l - r) < 1e-5; }));
}

TEST_CASE("Mul backward pass") {
  auto l = tensor::make<float>({1, 2, 3, 4});
  l.requires_grad(true);
  auto r = tensor::make<float>({6, 7, 8, 9});
  r.requires_grad(true);

  auto res = tensor::Mul<float>()(l, r);
  res.backward(tensor::make<float>({1, 2, 3, 4}));

  REQUIRE(l.grad().has_value());
  CHECK_THAT((*l.grad()).shape(), RangeEquals(std::vector<int>{4}));
  CHECK_THAT(*(*l.grad()).data(), RangeEquals(std::vector<float>{6, 14, 24, 36}));

  REQUIRE(r.grad().has_value());
  CHECK_THAT((*r.grad()).shape(), RangeEquals(std::vector<int>{4}));
  CHECK_THAT(*(*r.grad()).data(), RangeEquals(std::vector<float>{1, 4, 9, 16}));
}

TEST_CASE("Add backward pass") {
  auto l = tensor::make<float>({-9, -8, -7, -6});
  l.requires_grad(true);
  auto r = tensor::make<float>({6, 7, 8, 9});
  r.requires_grad(true);

  auto res = tensor::Add<float>()(l, r);
  res.backward(tensor::make<float>({1, 2, 3, 4}));

  REQUIRE(l.grad().has_value());
  CHECK_THAT((*l.grad()).shape(), RangeEquals(std::vector<int>{4}));
  CHECK_THAT(*(*l.grad()).data(), RangeEquals(std::vector<float>{1, 2, 3, 4}));

  REQUIRE(r.grad().has_value());
  CHECK_THAT((*r.grad()).shape(), RangeEquals(std::vector<int>{4}));
  CHECK_THAT(*(*r.grad()).data(), RangeEquals(std::vector<float>{1, 2, 3, 4}));
}

TEST_CASE("Sum default backward pass") {
  auto l = tensor::make<float>({5, 6, 7, 8});
  l.requires_grad(true);

  auto res = tensor::Sum<float>()(l, tensor::make<float>(0));
  res.backward();

  REQUIRE(l.grad().has_value());
  CHECK_THAT((*l.grad()).shape(), RangeEquals(std::vector<int>{4}));
  CHECK_THAT(*(*l.grad()).data(), RangeEquals(std::vector<float>{1, 1, 1, 1}));
}

TEST_CASE("Sum backward pass") {
  auto l = tensor::make<float>({5, 6, 7, 8});
  l.requires_grad(true);

  auto res = tensor::Sum<float>()(l, tensor::make<float>(0));
  res.backward(tensor::make<float>({10}));

  REQUIRE(l.grad().has_value());
  CHECK_THAT((*l.grad()).shape(), RangeEquals(std::vector<int>{4}));
  CHECK_THAT(*(*l.grad()).data(), RangeEquals(std::vector<float>{10, 10, 10, 10}));
}

TEST_CASE("MatMul backward pass") {
  auto l = tensor::make<float>({2, 2}, {5, 6, 7, 8});
  l.requires_grad(true);
  auto r = tensor::eye<float>(2);
  r.requires_grad(true);

  auto res = tensor::MatMul<float>()(l, r);
  res.backward(tensor::make<float>({2, 2}, {1, 1, 1, 1}));

  REQUIRE(l.grad().has_value());
  CHECK_THAT((*l.grad()).shape(), RangeEquals(std::vector<int>{2, 2}));
  CHECK_THAT(*(*l.grad()).data(), RangeEquals(std::vector<float>{1, 1, 1, 1}));

  REQUIRE(r.grad().has_value());
  CHECK_THAT((*r.grad()).shape(), RangeEquals(std::vector<int>{2, 2}));
  CHECK_THAT(*(*r.grad()).data(), RangeEquals(std::vector<float>{12, 12, 14, 14}));
}

TEST_CASE("Copy backward pass") {
  auto l = tensor::make<float>({5, 6, 7, 8});
  l.requires_grad(true);

  auto res = tensor::Copy<float>()(l);
  res.backward(tensor::make<float>({1, 2, 3, 4}));

  REQUIRE(l.grad().has_value());
  CHECK_THAT((*l.grad()).shape(), RangeEquals(std::vector<int>{4}));
  CHECK_THAT(*(*l.grad()).data(), RangeEquals(std::vector<float>{1, 2, 3, 4}));
}

TEST_CASE("View backward pass") {
  auto l = tensor::make<float>({5, 6, 7, 8});
  l.requires_grad(true);

  auto res = tensor::View<float>()(l, tensor::make<float>({2, 2}));
  res.backward(tensor::make<float>({2, 2}, {1, 2, 3, 4}));

  REQUIRE(l.grad().has_value());
  CHECK_THAT((*l.grad()).shape(), RangeEquals(std::vector<int>{4}));
  CHECK_THAT(*(*l.grad()).data(), RangeEquals(std::vector<float>{1, 2, 3, 4}));
}

TEST_CASE("Squeeze backward pass") {
  auto l = tensor::make<float>({4, 1, 1}, {5, 6, 7, 8});
  l.requires_grad(true);

  auto res = tensor::Squeeze<float>()(l);
  res.backward(tensor::make<float>({4}, {1, 2, 3, 4}));

  REQUIRE(l.grad().has_value());
  CHECK_THAT((*l.grad()).shape(), RangeEquals(std::vector<int>{4, 1, 1}));
  CHECK_THAT(*(*l.grad()).data(), RangeEquals(std::vector<float>{1, 2, 3, 4}));
}

TEST_CASE("Unsqueeze backward pass") {
  auto l = tensor::make<float>({2, 2}, {5, 6, 7, 8});
  l.requires_grad(true);

  auto res = tensor::Unsqueeze<float>()(l, tensor::make<float>(1));
  res.backward(tensor::make<float>({2, 1, 2}, {1, 2, 3, 4}));

  REQUIRE(l.grad().has_value());
  CHECK_THAT((*l.grad()).shape(), RangeEquals(std::vector<int>{2, 2}));
  CHECK_THAT(*(*l.grad()).data(), RangeEquals(std::vector<float>{1, 2, 3, 4}));
}
