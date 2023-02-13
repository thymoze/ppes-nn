#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <nn/all.hpp>
#include <tensor/tensor.hpp>

TEST_CASE("gradient_step_lossIsReduced") {
  auto model = nn::Sequential<float>();
  model.add(nn::Linear<float>(2, 2));
  model.add(nn::Sigmoid<float>());
  model.add(nn::Linear<float>(2, 1));
  model.init();
  auto optimizer = nn::SGD<float>(model.params(), 0.01);

  auto target = tensor::make<float>({1}, {1});
  auto train = tensor::make<float>({1, 2}, {2, 2});
  auto output = model.forward(train);

  auto loss_1 = nn::mse<float>(output, target);

  loss_1.backward();
  optimizer.step();

  output = model.forward(train);

  auto loss_2 = nn::mse<float>(output, target);

  REQUIRE(loss_1.item() > loss_2.item());
}
