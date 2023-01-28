#include <iostream>
#include <nn/dataset/mnist_dataset.hpp>
#include <nn/modules/linear.hpp>
#include <nn/modules/sigmoid.hpp>
#include <nn/mse.hpp>
#include <nn/optim/sgd.hpp>
#include <nn/sequential.hpp>
#include <tuple>

int main() {
  auto mnist = nn::MnistDataset("../../data", nn::MnistDataset::Set::TRAIN);

  auto model = nn::Sequential();
  model.add(nn::Linear(28 * 28, 800));
  model.add(nn::Sigmoid());
  model.add(nn::Linear(800, 10));

  auto optimizer = nn::SGD(model.params(), 0.01);

  for (int epoch = 0; epoch < 1000; epoch++) {
    double epoch_loss = 0;
    int i = 0;
    for (auto &[input, target] : mnist) {
      input.reshape(1, 28 * 28);
      auto output = model({input})[0];

      auto target_onehot = nn::Matrix<double>(1, 10, 0);
      target_onehot(0, static_cast<int>(target(0, 0))) = 1;

      auto loss = nn::mse(output, target_onehot);

      optimizer.zero_grad();
      loss.backward();

      optimizer.step();

      epoch_loss += loss(0, 0);
      std::cout << "\33[2K\r" << i << ": " << loss(0, 0) << std::flush;
      i++;
    }
    std::cout << "\33[2K\r";
    epoch_loss = epoch_loss / mnist.size();

    std::cout << "Epoch " << epoch << ": Loss = " << epoch_loss << std::endl;
  }
}
