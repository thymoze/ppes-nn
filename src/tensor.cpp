#include <algorithm>
#include <iostream>
#include <nn/dataset/mnist_dataset.hpp>
#include <nn/modules/linear.hpp>
#include <nn/modules/sigmoid.hpp>
#include <nn/mse.hpp>
#include <nn/optim/sgd.hpp>
#include <nn/sequential.hpp>
#include <tensor/tensor.hpp>
#include <vector>

int main() {
  auto mnist = nn::MnistDataset<double>("../../data", nn::MnistDataset<double>::Set::TRAIN);

  auto model = nn::Sequential<double>();
  model.add(nn::Linear<double>(28 * 28, 300));
  model.add(nn::Sigmoid<double>());
  model.add(nn::Linear<double>(300, 10));

  auto optimizer = nn::SGD<double>(model.params(), 0.0001);

  for (int epoch = 0; epoch < 1; epoch++) {
    double epoch_loss = 0;
    int i = 0;
    for (auto&& [input, target] : mnist) {
      optimizer.zero_grad();

      auto in = input.view({1, input.size()});
      auto output = model(in);

      auto target_onehot = Tensor<double>::zeros({1, 10});
      target_onehot(0, static_cast<std::size_t>(target.item())) = 1;
      auto loss = nn::mse(output, target_onehot);

      loss.backward();
      optimizer.step();

      epoch_loss += loss.item();
      std::cout << "\33[2K\r" << std::setw(3) << i++ << ": " << loss.item() << std::flush;
    }
    std::cout << "\33[2K\r";
    epoch_loss = epoch_loss / mnist.size();

    if (epoch % 100 == 0) {
      std::cout << "Epoch " << epoch << ": Loss = " << epoch_loss << std::endl;
    }
  }
}
