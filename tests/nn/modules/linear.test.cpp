#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <nn/all.hpp>
#include <tensor/tensor.hpp>

using Catch::Matchers::RangeEquals;

TEST_CASE("constructor") {
  auto linear = nn::Linear<float>(10, 100);

  REQUIRE(linear.is_linear() == true);
  REQUIRE(linear.num_in() == 10);
  REQUIRE(linear.num_out() == 100);
  REQUIRE(linear.bias() == true);
  REQUIRE(linear.params().size() == 2);
}

TEST_CASE("init") {
  auto linear = nn::Linear<float>(10, 10);

  linear.init();

  REQUIRE(linear.is_linear() == true);
  REQUIRE(linear.num_in() == 10);
  REQUIRE(linear.num_out() == 10);
  REQUIRE(linear.bias() == true);
  REQUIRE(linear.params().size() == 2);
  REQUIRE(linear.params()[0].template value<tensor::Tensor<float>>().size() == 100);
  CHECK_THAT(linear.params()[0].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<std::size_t>{10, 10}));
  REQUIRE(linear.params()[1].template value<tensor::Tensor<float>>().size() == 10);
  CHECK_THAT(linear.params()[1].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<std::size_t>{1, 10}));
}

TEST_CASE("init_withPretrainedWeights") {
  alignas(4) const unsigned char init_data[] = {0x0, 0x0, 0x80, 0x3f, 0x0, 0x0, 0x80, 0x3f,
                                                0x0, 0x0, 0x80, 0x3f, 0x0, 0x0, 0x80, 0x3f,
                                                0x0, 0x0, 0x80, 0x3f, 0x0, 0x0, 0x80, 0x3f};
  const unsigned int init_data_length = 24;

  auto linear = nn::Linear<float>(2, 2);
  linear.init(init_data, init_data_length);

  REQUIRE(linear.params()[0].template value<tensor::Tensor<float>>().size() == 4);
  CHECK_THAT(linear.params()[0].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<size_t>{2, 2}));
  CHECK_THAT(*linear.params()[0].template value<tensor::Tensor<float>>().data(),
             RangeEquals(std::vector<float>{1, 1, 1, 1}));

  REQUIRE(linear.params()[1].template value<tensor::Tensor<float>>().size() == 2);
  CHECK_THAT(linear.params()[1].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<std::size_t>{1, 2}));
  CHECK_THAT(*linear.params()[1].template value<tensor::Tensor<float>>().data(),
             RangeEquals(std::vector<float>{1, 1}));
}

TEST_CASE("forward") {
  // data for linear layer of size 2x2 with biases and data = {1, 1, 1, 1}
  alignas(4) const unsigned char init_data[] = {0x0, 0x0, 0x80, 0x3f, 0x0, 0x0, 0x80, 0x3f,
                                                0x0, 0x0, 0x80, 0x3f, 0x0, 0x0, 0x80, 0x3f,
                                                0x0, 0x0, 0x80, 0x3f, 0x0, 0x0, 0x80, 0x3f};
  const unsigned int init_data_length = 24;

  auto linear = nn::Linear<float>(2, 2);
  linear.init(init_data, init_data_length);

  auto result = linear.forward(tensor::make<float>({2, 2}, {2, 2, 2, 2}));

  // forward: input * weights + bias
  // [[2, 2]  *  [[1, 1]    + [[1]   = [[5, 5]
  //  [2, 2]]     [1, 1]]      [1]]     [5, 5]]
  CHECK_THAT(*result.data(), RangeEquals(std::vector<float>{5, 5, 5, 5}));
  CHECK_THAT(result.shape(), RangeEquals(std::vector<std::size_t>{2, 2}));
}

TEST_CASE("is_prunable_returnsTrue") {
  auto linear = nn::Linear<float>(10, 100);
  linear.init();

  auto result = linear.is_prunable();

  REQUIRE(result == true);
}

TEST_CASE("is_prunable_returnsFalse") {
  auto linear = nn::Linear<float>(1, 10);
  linear.init();

  auto result = linear.is_prunable();

  REQUIRE(result == false);
}

TEST_CASE("prune_one_neuron") {
  auto linear = nn::Linear<float>(10, 10);
  linear.init();

  auto abs_row_sums =
      tensor::min(tensor::abssum(linear.params()[0].template value<tensor::Tensor<float>>(), 0));
  int expected_neuron_to_prune = tensor::argmin(abs_row_sums).item();

  int actual_pruned_neuron = linear.prune_one_neuron();

  REQUIRE(expected_neuron_to_prune == actual_pruned_neuron);
  CHECK_THAT(linear.params()[0].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<std::size_t>{9, 10}));
  CHECK_THAT(linear.params()[1].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<std::size_t>{1, 10}));
}

TEST_CASE("apply_pruned_neuron") {
  auto linear = nn::Linear<float>(10, 10);
  linear.init();

  linear.apply_pruned_neuron(0);

  CHECK_THAT(linear.params()[0].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<std::size_t>{10, 9}));
  CHECK_THAT(linear.params()[1].template value<tensor::Tensor<float>>().shape(),
             RangeEquals(std::vector<std::size_t>{1, 9}));
}
