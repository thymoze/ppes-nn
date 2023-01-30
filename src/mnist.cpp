#include <algorithm>
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
  model.add(nn::Linear(28 * 28, 300));
  model.add(nn::Sigmoid());
  model.add(nn::Linear(300, 10));

  auto optimizer = nn::SGD(model.params(), 0.001);

  for (int epoch = 0; epoch < 100; epoch++) {
    double epoch_loss = 0;

    int i = 1;
    auto batch_loss = nn::Variable<double>(nn::Matrix<double>(1, 1));
    int batch_correct = 0;

    for (auto &[input, target] : mnist) {
      input.reshape(1, 28 * 28);
      auto output_onehot = model({input})[0];

      auto target_onehot = nn::Matrix<double>(1, 10, 0);
      target_onehot(0, static_cast<int>(target(0, 0))) = 1;

      auto output = std::distance(
          output_onehot.value().begin(),
          std::max_element(output_onehot.value().begin(), output_onehot.value().end()));
      if (output == target(0, 0)) {
        ++batch_correct;
      }

      auto loss = nn::mse(output_onehot, target_onehot);
      batch_loss = batch_loss + loss;
      epoch_loss += loss(0, 0);

      if (i % 50 == 0) {
        batch_loss.backward();

        optimizer.step();
        batch_loss.reset_dag();

        std::cout << "\33[2K\r" << i << ": "
                  << "batch_loss = " << batch_loss(0, 0)
                  << " batch_acc = " << static_cast<double>(batch_correct) / 50 << std::flush;
        batch_loss = nn::Variable<double>(nn::Matrix<double>(1, 1));
        batch_correct = 0;
      }

      i++;
    }
    std::cout << "\33[2K\r";
    epoch_loss = epoch_loss / mnist.size();

    std::cout << "Epoch " << epoch << ": Loss = " << epoch_loss << std::endl;
  }
}
