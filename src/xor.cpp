#include <iostream>
#include <nn/all.hpp>
#include <nn/quantization/quantization.hpp>

using tensor::Tensor;

int main() {
  nn::random::seed(0x5EED);

  std::pair<Tensor<float>, Tensor<float>> xor_data = {
      tensor::make<float>({4, 1, 2}, {0, 0, 1, 1, 1, 0, 0, 1}),
      tensor::make<float>({4, 1, 1}, {0, 0, 1, 1}),
  };

  auto model = nn::Sequential<float>();
  model.add(nn::Linear<float>(2, 3));
  model.add(nn::Sigmoid<float>());
  model.add(nn::Linear<float>(3, 1));
  model.init();

  auto optimizer = nn::SGD<float>(model.params(), 0.25);

  for (int epoch = 0; epoch < 2000; epoch++) {
    auto& [input, target] = xor_data;
    auto output = model(input);

    auto loss = nn::mse(output, target);

    optimizer.zero_grad();
    loss.backward();

    optimizer.step();

    if (epoch % 200 == 0) {
      std::cout << "\33[2K\r Epoch " << epoch << ": Loss = " << loss.item() << std::endl;
    }
  }

  model.save("../../trained_models/xor.hpp", "xor_model");
}
