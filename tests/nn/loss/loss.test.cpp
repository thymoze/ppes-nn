#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <nn/loss/loss.hpp>
#include <tensor/tensor.hpp>

using Catch::Matchers::RangeEquals;

TEST_CASE("loss") {
  auto pred = tensor::make<float>({1, 0});
  auto target = tensor::make<float>({1, 1});

  auto result = nn::mse<float>(pred, target);

  CHECK_THAT(*result.data(), RangeEquals(std::vector<float>{0.5}));
}

TEST_CASE("loss_targetAndPredIsEqual_ReturnsZero") {
  std::vector<float> target_data(100);
  std::vector<float> pred_data(100);

  std::iota(target_data.begin(), target_data.end(), 1);
  std::iota(pred_data.begin(), pred_data.end(), 1);

  auto target = tensor::make<float>({100}, std::move(target_data));
  auto pred = tensor::make<float>({100}, std::move(pred_data));

  auto result = nn::mse<float>(pred, target);

  CHECK_THAT(*result.data(), RangeEquals(std::vector<float>{0}));
}

TEST_CASE("loss_invalidInput") {
  auto target = tensor::rand<float>(tensor::Shape({10}));
  auto pred = tensor::rand<float>(tensor::Shape({20}));

  CHECK_THROWS_AS(nn::mse<float>(pred, target), std::logic_error);
}
