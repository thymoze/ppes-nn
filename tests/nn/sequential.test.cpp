#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <nn/all.hpp>
#include <tensor/tensor.hpp>

using Catch::Matchers::RangeEquals;

TEST_CASE("add") {
  auto [mod, expected_params] = GENERATE(std::pair{nn::Linear<float>(2, 2), 2},
                                         std::pair{nn::Linear<float>(10, 10, false), 1});

  auto model = nn::Sequential<float>();
  model.add(mod);

  REQUIRE(model.params().size() == expected_params);
}

TEST_CASE("add_multiple_layer") {
  auto [mods, expected_params] = GENERATE(
      std::pair{std::vector{nn::Linear<float>(2, 2), nn::Linear<float>(2, 4)}, 4},
      std::pair{std::vector{nn::Linear<float>(2, 2, false), nn::Linear<float>(2, 4)}, 3},
      std::pair{std::vector{nn::Linear<float>(2, 2, false), nn::Linear<float>(2, 4, false)}, 2});

  auto model = nn::Sequential<float>();
  for (auto &m : mods) {
    model.add(m);
  }

  REQUIRE(model.params().size() == expected_params);
}

TEST_CASE("init") {
  auto model = nn::Sequential<float>();
  model.add(nn::Linear<float>(2, 2));
  model.add(nn::Sigmoid<float>());
  model.add(nn::Linear<float>(2, 4));

  model.init();

  REQUIRE(model.params().size() == 4);
  REQUIRE(model.params()[0].template value<tensor::Tensor<float>>().size() == 4);
  CHECK_THAT(model.params()[0].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<std::size_t>{2, 2}));
  REQUIRE(model.params()[1].template value<tensor::Tensor<float>>().size() == 2);
  CHECK_THAT(model.params()[1].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<std::size_t>{1, 2}));
  REQUIRE(model.params()[2].template value<tensor::Tensor<float>>().size() == 8);
  CHECK_THAT(model.params()[2].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<std::size_t>{2, 4}));
  REQUIRE(model.params()[3].template value<tensor::Tensor<float>>().size() == 4);
  CHECK_THAT(model.params()[3].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<std::size_t>{1, 4}));
}

TEST_CASE("forward") {
  auto model = nn::Sequential<float>();
  model.add(nn::Linear<float>(2, 2));
  model.add(nn::Sigmoid<float>());
  model.add(nn::Linear<float>(2, 4));
  model.add(nn::ReLU<float>());
  model.add(nn::Linear<float>(4, 16));
  model.init();

  auto input = tensor::make<float>({1, 2}, {4, 4});

  auto result = model.forward(input);

  REQUIRE(result.size() == 16);
  CHECK_THAT(result.shape(), RangeEquals(std::vector<std::size_t>{1, 16}));
}

TEST_CASE("prune_one_neuron_correctLayerIsPruned") {
  auto model = nn::Sequential<float>();
  model.add(nn::Linear<float>(2, 2));
  model.add(nn::Sigmoid<float>());
  model.add(nn::Linear<float>(2, 2));
  model.init();

  model.prune_one_neuron();

  CHECK_THAT(model.modules()[0].get()->params()[0].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<std::size_t>{2, 1}));
  CHECK_THAT(model.modules()[0].get()->params()[1].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<std::size_t>{1, 1}));
  CHECK_THAT(model.modules()[2].get()->params()[0].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<std::size_t>{1, 2}));
  CHECK_THAT(model.modules()[2].get()->params()[1].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<std::size_t>{1, 2}));
}
