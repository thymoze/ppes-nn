#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <tensor/tensor.hpp>

using Catch::Matchers::RangeEquals;
using namespace tensor;

TEST_CASE("Element access") {
  auto data = std::make_shared<std::vector<int>>(2 * 3 * 4);
  std::iota(data->begin(), data->end(), 0);
  auto t = VectorStorage<int>(data, {2, 3, 4});

  auto v = t.at(5);
  CHECK(v == 5);

  t.at(5) = 100;
  CHECK(t.get({0, 1, 1}) == 100);
  CHECK(t.get({1, 1, 1}) == 17);

  CHECK(*t.begin() == 0);
  CHECK(*(t.begin() + 10) == 10);
  CHECK(*(t.end() - 10) == 14);
  CHECK(*(t.end() - 1) == 23);

  auto t2 = Tensor<int>(std::make_unique<VectorStorage<int>>(std::move(t)));
  CHECK(t2(1, 2, 3) == 23);
  CHECK(t2[{1, 0, 3}] == 15);
}

TEST_CASE("Reshaping") {
  auto data = std::make_shared<std::vector<int>>(2 * 3 * 4);
  std::iota(data->begin(), data->end(), 0);
  auto t = VectorStorage<int>(data, {2, 3, 4});

  auto t2 = t.clone();
  CHECK(t.begin() == t2->begin());
  CHECK_THAT(t.shape(), RangeEquals(t2->shape()));
  CHECK_THAT(t.strides(), RangeEquals(t2->strides()));

  auto t3 = t.view({2, 12});
  CHECK(t.begin() == t3->begin());
  CHECK_THAT(t3->shape(), RangeEquals(std::vector<int>{2, 12}));
  CHECK_THAT(t3->strides(), RangeEquals(std::vector<int>{12, 1}));

  auto t4 = t.view(Shape{6, 4}, Strides{4, 1});
  CHECK(t.begin() == t4->begin());
  CHECK_THAT(t4->shape(), RangeEquals(std::vector<int>{6, 4}));
  CHECK_THAT(t4->strides(), RangeEquals(std::vector<int>{4, 1}));

  auto t5 = t.permute({1, 2, 0});
  CHECK(t.begin() == t5->begin());
  CHECK_THAT(t5->shape(), RangeEquals(std::vector<int>{3, 4, 2}));
  CHECK_THAT(t5->strides(), RangeEquals(std::vector<int>{4, 1, 12}));
}

TEST_CASE("Remove 1D") {
  auto t = tensor::make<float>({2, 3, 4});

  INFO("Input:\n" << t);

  auto res = t.remove(0, 1);
  INFO("Removed (0, 1):\n" << res);
  CHECK_THAT(res.shape(), RangeEquals(std::vector<int>{2}));
  CHECK_THAT(*res.data(), RangeEquals(std::vector<float>{2, 4}));
}

TEST_CASE("Remove ND") {
  std::vector<float> data(2 * 3 * 4);
  std::iota(data.begin(), data.end(), 1);
  auto t = tensor::make<float>({2, 3, 4}, std::move(data));

  INFO("Input:\n" << t);

  auto res = t.remove(2, 2);
  INFO("Removed (2, 2):\n" << res);
  CHECK_THAT(res.shape(), RangeEquals(std::vector<int>{2, 3, 3}));
  CHECK_THAT(*res.data(), RangeEquals(std::vector<float>{1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16,
                                                         17, 18, 20, 21, 22, 24}));

  auto res2 = res.remove(0, 1);
  INFO("Removed (0, 1):\n" << res2);
  CHECK_THAT(res2.shape(), RangeEquals(std::vector<int>{1, 3, 3}));
  CHECK_THAT(*res2.data(), RangeEquals(std::vector<float>{1, 2, 4, 5, 6, 8, 9, 10, 12}));

  auto res3 = res2.remove(0, 0);
  INFO("Removed (0, 1):\n" << res3);
  CHECK_THAT(res3.shape(), RangeEquals(std::vector<int>{0, 3, 3}));
  CHECK_THAT(*res3.data(), RangeEquals(std::vector<float>{}));

  CHECK_THROWS_AS(res3.remove(0, 1), std::logic_error);
}
