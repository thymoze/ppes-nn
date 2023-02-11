#include <algorithm>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <matrix/matrix.hpp>
#include <random>
#include <vector>

using namespace nn;

TEST_CASE("Initialize Matrix of requested size") {
  auto m = Matrix<int>(5, 3);

  REQUIRE(m.rows() == 5);
  REQUIRE(m.cols() == 3);
  REQUIRE(m.data().size() == 5 * 3);
  REQUIRE(std::all_of(m.begin(), m.end(), [](auto& val) { return val == 0; }));
}

TEST_CASE("Initialize Matrix with number") {
  auto m = Matrix<int>(2, 6, 12);

  REQUIRE(m.rows() == 2);
  REQUIRE(m.cols() == 6);
  REQUIRE(m.data().size() == 2 * 6);
  REQUIRE(std::all_of(m.begin(), m.end(), [](auto& val) { return val == 12; }));
}

TEST_CASE("Initialize Matrix with initialization list") {
  auto m = Matrix<int>({{1, 2, 3, 4}, {4, 5, 6, 7}});

  REQUIRE(m.rows() == 2);
  REQUIRE(m.cols() == 4);
  REQUIRE(m.data().size() == 2 * 4);
  auto v = std::vector<int>{{1, 2, 3, 4, 4, 5, 6, 7}};
  REQUIRE(std::equal(m.begin(), m.end(), v.begin()));
}

TEST_CASE("matmul") {
  auto m1 = Matrix<int>({{1, 2}, {5, 6}});
  auto m2 = Matrix<int>({{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto expected = Matrix<int>({{11, 14, 17, 20}, {35, 46, 57, 68}});

  auto result = m1.matmul(m2);

  REQUIRE(std::equal(result.begin(), result.end(), expected.begin()));
}

TEST_CASE("matmul_benchmark") {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<double> dist{1, 52};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

  std::vector<double> vec(50 * 50);
  std::generate(vec.begin(), vec.end(), gen);
  auto m1 = Matrix<double>(50, 50, std::move(vec));

  std::vector<double> vec2(50 * 50);
  std::generate(vec2.begin(), vec2.end(), gen);
  auto m2 = Matrix<double>(50, 50, std::move(vec2));

  // BENCHMARK("matmul parallel") { return m1.matmul(m2); };
  // BENCHMARK("matmul sequential") { return m1.matmul_seq(m2); };
}

TEST_CASE("delete_row") {
  auto m = Matrix<int>({{1, 2}, {5, 6}});
  auto expected = Matrix<int>({{5, 6}});

  m.delete_row(0);

  REQUIRE(m.rows() == 1);
  REQUIRE(m.cols() == 2);
  REQUIRE(std::equal(m.begin(), m.end(), expected.begin()));
  REQUIRE(m.data().size() == expected.data().size());
}

TEST_CASE("delete_column") {
  auto m = Matrix<int>({{1, 2, 3}, {3, 4, 5}});
  auto expected = Matrix<int>({{1, 3}, {3, 5}});

  m.delete_column(1);

  REQUIRE(m.rows() == 2);
  REQUIRE(m.cols() == 2);
  REQUIRE(std::equal(m.begin(), m.end(), expected.begin()));
  REQUIRE(m.data().size() == expected.data().size());
}

TEST_CASE("lowest_row_sum") {
  auto m = Matrix<int>({{1, 1, 1}, {2, 2, 2}, {3, 3, 3}});
  int expected = 0;

  auto result = m.lowest_row_sum();

  REQUIRE(result == expected);
}
