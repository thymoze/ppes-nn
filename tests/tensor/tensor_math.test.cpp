#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <tensor/tensor.hpp>

using Catch::Matchers::AllMatch;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::RangeEquals;
using Catch::Matchers::WithinRel;

TEST_CASE("Negation") {
  auto l = tensor::make<int>({2, 3}, {1, 2, 3, 4, 5, 6});

  auto res = -l;
  CHECK_THAT(res.shape(), RangeEquals(std::vector<int>{2, 3}));
  CHECK_THAT(*res.data(), RangeEquals(std::vector<int>{-1, -2, -3, -4, -5, -6}));
}

TEST_CASE("Simple addition") {
  auto l = tensor::make<int>({2, 3}, {1, 2, 3, 4, 5, 6});
  auto r = tensor::make<int>({2, 3}, {1, 2, 3, 4, 5, 6});

  auto res = l + r;
  CHECK_THAT(res.shape(), RangeEquals(std::vector<int>{2, 3}));
  CHECK_THAT(*res.data(), RangeEquals(std::vector<int>{2, 4, 6, 8, 10, 12}));
}

TEST_CASE("Broadcasting addition") {
  auto l = tensor::make<int>({4, 1}, {1, 2, 3, 4});
  auto r = tensor::make<int>({1, 4}, {1, 2, 3, 4});

  auto res = l + r;
  CHECK_THAT(res.shape(), RangeEquals(std::vector<int>{4, 4}));
  CHECK_THAT(*res.data(),
             RangeEquals(std::vector<int>{2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8}));

  auto r2 = tensor::make<int>(10);
  auto res2 = l + r2;
  CHECK_THAT(res2.shape(), RangeEquals(std::vector<int>{4, 1}));
  CHECK_THAT(*res2.data(), RangeEquals(std::vector<int>{11, 12, 13, 14}));
}

TEST_CASE("Simple subtraction") {
  auto l = tensor::make<int>({2, 3}, {1, 2, 3, 4, 5, 6});
  auto r = tensor::make<int>({2, 3}, {1, 2, 3, 4, 5, 6});

  auto res = l - r;
  CHECK_THAT(res.shape(), RangeEquals(std::vector<int>{2, 3}));
  CHECK_THAT(*res.data(), RangeEquals(std::vector<int>{0, 0, 0, 0, 0, 0}));
}

TEST_CASE("Broadcasting subtraction") {
  auto l = tensor::make<int>({4, 1}, {1, 2, 3, 4});
  auto r = tensor::make<int>({1, 4}, {1, 2, 3, 4});

  auto res = l - r;
  CHECK_THAT(res.shape(), RangeEquals(std::vector<int>{4, 4}));
  CHECK_THAT(*res.data(),
             RangeEquals(std::vector<int>{0, -1, -2, -3, 1, 0, -1, -2, 2, 1, 0, -1, 3, 2, 1, 0}));

  auto r2 = tensor::make<int>(10);
  auto res2 = l - r2;
  CHECK_THAT(res2.shape(), RangeEquals(std::vector<int>{4, 1}));
  CHECK_THAT(*res2.data(), RangeEquals(std::vector<int>{-9, -8, -7, -6}));
}

TEST_CASE("Simple multiplication") {
  auto l = tensor::make<int>({2, 3}, {1, 2, 3, 4, 5, 6});
  auto r = tensor::make<int>({2, 3}, {1, 2, 3, 4, 5, 6});

  auto res = l * r;
  CHECK_THAT(res.shape(), RangeEquals(std::vector<int>{2, 3}));
  CHECK_THAT(*res.data(), RangeEquals(std::vector<int>{1, 4, 9, 16, 25, 36}));
}

TEST_CASE("Broadcasting multiplication") {
  auto l = tensor::make<int>({4, 1}, {1, 2, 3, 4});
  auto r = tensor::make<int>({1, 4}, {1, 2, 3, 4});

  auto res = l * r;
  CHECK_THAT(res.shape(), RangeEquals(std::vector<int>{4, 4}));
  CHECK_THAT(*res.data(),
             RangeEquals(std::vector<int>{1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12, 4, 8, 12, 16}));

  auto r2 = tensor::make<int>(10);
  auto res2 = l * r2;
  CHECK_THAT(res2.shape(), RangeEquals(std::vector<int>{4, 1}));
  CHECK_THAT(*res2.data(), RangeEquals(std::vector<int>{10, 20, 30, 40}));
}

TEST_CASE("Simple division") {
  auto l = tensor::make<float>({2, 3}, {2, 4, 9, 16, 25, 36});
  auto r = tensor::make<float>({2, 3}, {1, 2, 3, 4, 5, 6});

  auto res = l / r;
  CHECK_THAT(res.shape(), RangeEquals(std::vector<int>{2, 3}));
  CHECK_THAT(*res.data(), RangeEquals(std::vector<float>{2, 2, 3, 4, 5, 6}));
}

TEST_CASE("Broadcasting division") {
  auto l = tensor::make<float>({4, 1}, {1., 2., 3., 4.});
  auto r = tensor::make<float>({1, 4}, {1., 2., 3., 4.});

  auto res = l / r;
  CHECK_THAT(res.shape(), RangeEquals(std::vector<float>{4, 4}));
  CHECK_THAT(*res.data(),
             RangeEquals(std::vector<float>{1, 1. / 2, 1. / 3, 1. / 4, 2, 1, 2. / 3, 2. / 4, 3,
                                            3. / 2, 1, 3. / 4, 4, 4. / 2, 4. / 3, 1}));

  auto r2 = tensor::make<float>(10);
  auto res2 = l / r2;
  CHECK_THAT(res2.shape(), RangeEquals(std::vector<int>{4, 1}));
  CHECK_THAT(*res2.data(), RangeEquals(std::vector<float>{1. / 10, 2. / 10, 3. / 10, 4. / 10}));
}

TEST_CASE("Sum") {
  std::vector<int> data(2 * 3 * 4);
  std::iota(data.begin(), data.end(), 1);
  auto t = tensor::make<int>({2, 3, 4}, std::move(data));

  auto sum_all = tensor::sum(t);
  CHECK_THAT(sum_all.shape(), RangeEquals(std::vector<int>{1}));
  CHECK_THAT(*sum_all.data(), RangeEquals(std::vector<int>{300}));

  auto sum_0 = tensor::sum(t, 0);
  CHECK_THAT(sum_0.shape(), RangeEquals(std::vector<int>{1, 3, 4}));
  CHECK_THAT(*sum_0.data(),
             RangeEquals(std::vector<int>{14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36}));

  auto sum_1 = tensor::sum(t, 1);
  CHECK_THAT(sum_1.shape(), RangeEquals(std::vector<int>{2, 1, 4}));
  CHECK_THAT(*sum_1.data(), RangeEquals(std::vector<int>{15, 18, 21, 24, 51, 54, 57, 60}));

  auto sum_2 = tensor::sum(t, 2);
  CHECK_THAT(sum_2.shape(), RangeEquals(std::vector<int>{2, 3, 1}));
  CHECK_THAT(*sum_2.data(), RangeEquals(std::vector<int>{10, 26, 42, 58, 74, 90}));
}

TEST_CASE("Absolute Sum") {
  auto t = tensor::make<float>({2, 3}, {-4, 1, -9, 2, -1, 2});

  auto sum_all = tensor::abssum(t);
  CHECK_THAT(sum_all.shape(), RangeEquals(std::vector<int>{1}));
  CHECK_THAT(*sum_all.data(), RangeEquals(std::vector<float>{19}));

  auto sum_0 = tensor::abssum(t, 0);
  CHECK_THAT(sum_0.shape(), RangeEquals(std::vector<float>{1, 3}));
  CHECK_THAT(*sum_0.data(), RangeEquals(std::vector<float>{6, 2, 11}));

  auto sum_1 = tensor::abssum(t, 1);
  CHECK_THAT(sum_1.shape(), RangeEquals(std::vector<int>{2, 1}));
  CHECK_THAT(*sum_1.data(), RangeEquals(std::vector<float>{14, 5}));
}

TEST_CASE("Mean") {
  std::vector<float> data(2 * 3 * 4);
  std::iota(data.begin(), data.end(), 1);
  auto t = tensor::make<float>({2, 3, 4}, std::move(data));

  auto mean_all = tensor::mean(t);
  CHECK_THAT(mean_all.shape(), RangeEquals(std::vector<int>{1}));
  CHECK_THAT(*mean_all.data(), RangeEquals(std::vector<float>{12.5}));

  auto mean_0 = tensor::mean(t, 0);
  CHECK_THAT(mean_0.shape(), RangeEquals(std::vector<float>{1, 3, 4}));
  CHECK_THAT(*mean_0.data(),
             RangeEquals(std::vector<float>{7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}));

  auto mean_1 = tensor::mean(t, 1);
  CHECK_THAT(mean_1.shape(), RangeEquals(std::vector<int>{2, 1, 4}));
  CHECK_THAT(*mean_1.data(), RangeEquals(std::vector<float>{5, 6, 7, 8, 17, 18, 19, 20}));

  auto mean_2 = tensor::mean(t, 2);
  CHECK_THAT(mean_2.shape(), RangeEquals(std::vector<int>{2, 3, 1}));
  CHECK_THAT(*mean_2.data(), RangeEquals(std::vector<float>{2.5, 6.5, 10.5, 14.5, 18.5, 22.5}));
}

TEST_CASE("ReLU") {
  auto t = tensor::make<float>({5, 1}, {5, 0, 0.1, -0.1, -3});

  auto res = tensor::relu(t);
  CHECK_THAT(res.shape(), RangeEquals(std::vector<int>{5, 1}));
  CHECK_THAT(*res.data(), RangeEquals(std::vector<float>{5, 0, 0.1, 0, 0}));
}

TEST_CASE("Sigmoid") {
  auto t = tensor::make<float>({1, 7}, {-10, -1, -0.5, 0, 0.5, 1, 10});

  auto res = tensor::sigmoid(t);
  CHECK_THAT(res.shape(), RangeEquals(std::vector<int>{1, 7}));
  CHECK_THAT(*res.data(), RangeEquals(std::vector<float>{4.5398e-05, 0.26894, 0.37754, 0.50000,
                                                         0.62246, 0.73106, 0.99995},
                                      [](auto l, auto r) { return std::abs(l - r) < 1e-5; }));
}

TEST_CASE("Softmax") {
  auto t = tensor::rand<float>({2, 3, 4});

  auto softmax_0 = tensor::softmax(t, 0);
  CHECK_THAT(softmax_0.shape(), RangeEquals(std::vector<int>{2, 3, 4}));
  CHECK_THAT(*tensor::sum(softmax_0, 0).data(), AllMatch(WithinRel(1, 1e-6)));

  auto softmax_1 = tensor::softmax(t, 1);
  CHECK_THAT(softmax_1.shape(), RangeEquals(std::vector<int>{2, 3, 4}));
  CHECK_THAT(*tensor::sum(softmax_1, 1).data(), AllMatch(WithinRel(1, 1e-6)));

  auto softmax_2 = tensor::softmax(t, 2);
  CHECK_THAT(softmax_2.shape(), RangeEquals(std::vector<int>{2, 3, 4}));
  CHECK_THAT(*tensor::sum(softmax_2, 2).data(), AllMatch(WithinRel(1, 1e-6)));
}

TEST_CASE("2D Matrix multiplication") {
  auto ops = GENERATE(tensor::SimpleOps<float>(), tensor::MTOps<float>());
  auto backend = tensor::TensorBackend<float>(ops);

  std::vector<float> data(2 * 3);
  std::iota(data.begin(), data.end(), 1);
  auto l = tensor::make<float>({2, 3}, std::vector<float>(data), backend);
  auto r = tensor::eye<float>(3, backend);

  auto res = tensor::matmul(l, r);
  CHECK_THAT(res.shape(), RangeEquals(std::vector<int>{2, 3}));
  CHECK_THAT(*res.data(), RangeEquals(std::vector<float>{1, 2, 3, 4, 5, 6}));

  auto r2 = tensor::make<float>({3, 4}, {1, -2, 3, -4, 5, -6, 6, -5, 4, -3, 2, -1});
  auto res2 = tensor::matmul(l, r2);

  CHECK_THAT(res2.shape(), RangeEquals(std::vector<int>{2, 4}));
  CHECK_THAT(*res2.data(), RangeEquals(std::vector<float>{23, -23, 21, -17, 53, -56, 54, -47}));
}

TEST_CASE("Broadcasting matrix multiplication") {
  auto ops = GENERATE(tensor::SimpleOps<float>(), tensor::MTOps<float>());
  auto backend = tensor::TensorBackend<float>(ops);

  std::vector<float> data(4 * 1 * 2 * 3);
  std::iota(data.begin(), data.end(), 1);
  auto l = tensor::make<float>({4, 1, 2, 3}, std::vector<float>(data), backend);
  auto r = tensor::Tensor<float>(l.data()->permute({1, 0, 3, 2}), backend);

  auto res = tensor::matmul(l, r);
  CHECK_THAT(res.shape(), RangeEquals(std::vector<int>{4, 4, 2, 2}));
  CHECK(tensor::sum(res).item() == 30128);

  auto r2 = tensor::zeros<float>({5, 5});
  CHECK_THROWS_WITH(tensor::matmul(l, r2), ContainsSubstring("dimensions don't match"));
}

TEST_CASE("Maximum") {
  auto t = tensor::make<float>({2, 3}, {4, -1, 9, 2, -2, 2});

  auto max_all = tensor::max(t);
  CHECK_THAT(max_all.shape(), RangeEquals(std::vector<int>{1}));
  CHECK_THAT(*max_all.data(), RangeEquals(std::vector<float>{9}));

  auto max_0 = tensor::max(t, 0);
  CHECK_THAT(max_0.shape(), RangeEquals(std::vector<float>{1, 3}));
  CHECK_THAT(*max_0.data(), RangeEquals(std::vector<float>{4, -1, 9}));

  auto max_1 = tensor::max(t, 1);
  CHECK_THAT(max_1.shape(), RangeEquals(std::vector<int>{2, 1}));
  CHECK_THAT(*max_1.data(), RangeEquals(std::vector<float>{9, 2}));
}

TEST_CASE("Minimum") {
  auto t = tensor::make<float>({2, 3}, {4, 1, 9, 2, -1, 2});

  auto min_all = tensor::min(t);
  CHECK_THAT(min_all.shape(), RangeEquals(std::vector<int>{1}));
  CHECK_THAT(*min_all.data(), RangeEquals(std::vector<float>{-1}));

  auto min_0 = tensor::min(t, 0);
  CHECK_THAT(min_0.shape(), RangeEquals(std::vector<float>{1, 3}));
  CHECK_THAT(*min_0.data(), RangeEquals(std::vector<float>{2, -1, 2}));

  auto min_1 = tensor::min(t, 1);
  CHECK_THAT(min_1.shape(), RangeEquals(std::vector<int>{2, 1}));
  CHECK_THAT(*min_1.data(), RangeEquals(std::vector<float>{1, -1}));
}

TEST_CASE("Argmax") {
  auto t = tensor::make<float>({2, 3}, {5, -3, 3, -4, -2, 7});

  auto max_all = tensor::argmax(t);
  CHECK_THAT(max_all.shape(), RangeEquals(std::vector<int>{1}));
  CHECK_THAT(*max_all.data(), RangeEquals(std::vector<float>{5}));

  auto max_0 = tensor::argmax(t, 0);
  CHECK_THAT(max_0.shape(), RangeEquals(std::vector<float>{1, 3}));
  CHECK_THAT(*max_0.data(), RangeEquals(std::vector<float>{0, 1, 1}));

  auto max_1 = tensor::argmax(t, 1);
  CHECK_THAT(max_1.shape(), RangeEquals(std::vector<int>{2, 1}));
  CHECK_THAT(*max_1.data(), RangeEquals(std::vector<float>{0, 2}));
}

TEST_CASE("Argmin") {
  auto t = tensor::make<float>({2, 3}, {5, -3, 3, -4, -2, 7});

  auto min_all = tensor::argmin(t);
  CHECK_THAT(min_all.shape(), RangeEquals(std::vector<int>{1}));
  CHECK_THAT(*min_all.data(), RangeEquals(std::vector<float>{3}));

  auto min_0 = tensor::argmin(t, 0);
  CHECK_THAT(min_0.shape(), RangeEquals(std::vector<float>{1, 3}));
  CHECK_THAT(*min_0.data(), RangeEquals(std::vector<float>{1, 0, 0}));

  auto min_1 = tensor::argmin(t, 1);
  CHECK_THAT(min_1.shape(), RangeEquals(std::vector<int>{2, 1}));
  CHECK_THAT(*min_1.data(), RangeEquals(std::vector<float>{1, 0}));
}
