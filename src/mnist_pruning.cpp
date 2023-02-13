#define DEFAULT_TENSOR_BACKEND TensorBackend<T>(MTOps<T>())

#include <algorithm>
#include <iostream>
#include <nn/all.hpp>
#include <nn/dataset/mnist_dataset.hpp>
#include <nn/quantization/quantization.hpp>
#include <vector>

#include "train.hpp"

using tensor::Tensor;

int main() {
  auto mnist = nn::MnistDataset<float>("../../data", nn::MnistDataset<float>::Set::TRAIN, 100);

  // without pruning
  auto model = nn::Sequential<float>();
  model.add(nn::Linear<float>(28 * 28, 20));
  model.add(nn::Sigmoid<float>());
  model.add(nn::Linear<float>(20, 10));
  model.init();

  auto optimizer = nn::SGD<float>(model.params(), 0.01);

  train(model, 100, 3, 0.01);

  // with pruning
  auto model2 = nn::Sequential<float>();
  model2.add(nn::Linear<float>(28 * 28, 100));
  model2.add(nn::Sigmoid<float>());
  model2.add(nn::Linear<float>(100, 10));
  model2.init();

  auto optimizer2 = nn::SGD<float>(model2.params(), 0.01);
  train(model2, 100, 1, 0.01);

  model2.prune(80);

  train(model2, 100, 2, 0.01);

  std::cout << "Evaluation of model without pruning" << std::endl;
  evaluate<float>(model);
  std::cout << "Evaluation of model with pruning" << std::endl;
  evaluate<float>(model2);
}
