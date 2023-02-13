#define DEFAULT_TENSOR_BACKEND TensorBackend<T>(MTOps<T>())

#include <algorithm>
#include <iostream>
#include <nn/all.hpp>
#include <nn/dataset/mnist_dataset.hpp>
#include <nn/quantization/quantization.hpp>
#include <vector>

using tensor::Tensor;

template <typename T>
void evaluate(nn::Sequential<T>& model) {
  std::cout << "Evaluating..." << std::endl;

  auto test = nn::MnistDataset<float>("../../data", nn::MnistDataset<float>::Set::TEST, 100);
  float correct = 0;
  for (auto&& [input, target] : test) {
    auto in = input.view({input.shape()[0], 28 * 28});
    auto output_onehot = model(in);

    auto output = tensor::argmax(output_onehot, 1);
    correct +=
        tensor::sum(output.view({input.shape()[0]}) == target.view({input.shape()[0]})).item();
  }
  correct /= test.count();
  std::cout << "Test accuracy = " << correct << std::endl;
}

void train_one_epoch(nn::MnistDataset<float> mnist, nn::SGD<float>& optimizer,
                     nn::Sequential<float>& model, int epoch) {
  float epoch_loss = 0;
  int i = 0;
  for (auto&& [input, target] : mnist) {
    optimizer.zero_grad();

    auto in = input.view({input.shape()[0], 28 * 28});
    auto output_onehot = model(in);

    auto output = tensor::argmax(output_onehot, 1);
    auto batch_correct =
        tensor::mean(output.view({input.shape()[0]}) == target.view({input.shape()[0]}));

    auto target_onehot = tensor::zeros<float>({input.shape()[0], 10});
    for (std::size_t d = 0; d < input.shape()[0]; ++d) {
      target_onehot(d, static_cast<std::size_t>(target(d, 0, 0))) = 1;
    }
    auto loss = nn::mse(output_onehot, target_onehot);

    loss.backward();
    optimizer.step();

    epoch_loss += loss.item();
    i += mnist.batch_size();
    std::cout << "\33[2K\r" << std::setw(3) << i << ": loss = " << loss.item()
              << " correct = " << batch_correct.item() << std::flush;
    // model.save("../../trained_models/mnist_float_100.hpp", "mnist_model");
  }
  std::cout << "\33[2K\r";
  epoch_loss = epoch_loss / mnist.size();

  std::cout << "Epoch " << epoch << ": Loss = " << epoch_loss << std::endl;
}

int main() {
  auto mnist = nn::MnistDataset<float>("../../data", nn::MnistDataset<float>::Set::TRAIN, 100);

  // without pruning
  auto model = nn::Sequential<float>();
  model.add(nn::Linear<float>(28 * 28, 20));
  model.add(nn::Sigmoid<float>());
  model.add(nn::Linear<float>(20, 10));
  model.init();

  auto optimizer = nn::SGD<float>(model.params(), 0.01);

  for (int epoch = 0; epoch < 3; ++epoch) {
    train_one_epoch(mnist, optimizer, model, epoch);
  }

  // with pruning
  auto model2 = nn::Sequential<float>();
  model2.add(nn::Linear<float>(28 * 28, 100));
  model2.add(nn::Sigmoid<float>());
  model2.add(nn::Linear<float>(100, 10));
  model2.init();

  auto optimizer2 = nn::SGD<float>(model2.params(), 0.01);

  train_one_epoch(mnist, optimizer2, model2, 0);

  model2.prune(80);

  train_one_epoch(mnist, optimizer2, model2, 1);
  train_one_epoch(mnist, optimizer2, model2, 2);

  std::cout << "Evaluation of model without pruning" << std::endl;
  evaluate<float>(model);
  std::cout << "Evaluation of model with pruning" << std::endl;
  evaluate<float>(model2);
}