#include <iostream>
#include <nn/modules/linear.hpp>
#include <nn/modules/sigmoid.hpp>
#include <nn/mse.hpp>
#include <nn/optim/sgd.hpp>
#include <nn/sequential.hpp>

int main() {
  nn::random::seed(0x5EED);

  std::pair<Tensor<double>, Tensor<double>> xor_data = {
      Tensor<double>::make({4, 1, 2}, {0, 0, 1, 1, 1, 0, 0, 1}),
      Tensor<double>::make({4, 1, 1},    {   0,    0,    1,    1}),
  };

  auto model = nn::Sequential<double>();
  model.add(nn::Linear<double>(2, 3));
  model.add(nn::Sigmoid<double>());
  model.add(nn::Linear<double>(3, 1));

  auto optimizer = nn::SGD(model.params(), 0.25);

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

  //model.save("XOR_Model");
}
