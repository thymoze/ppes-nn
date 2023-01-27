#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <matrix/matrix.hpp>
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
