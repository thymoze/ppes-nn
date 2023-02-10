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

int main() {
  nn::random::seed(0x5EED);

  auto mnist = nn::MnistDataset<float>("../../data", nn::MnistDataset<float>::Set::TRAIN, 100);

  auto model = nn::Sequential<float>();
  model.add(nn::Linear<float>(28 * 28, 100));
  model.add(nn::Sigmoid<float>());
  model.add(nn::Linear<float>(100, 10));
  model.init();

  auto optimizer = nn::SGD<float>(model.params(), 0.01);

  for (int epoch = 0; epoch < 1; epoch++) {
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
      model.save("../../trained_models/mnist_float_100.hpp", "mnist_model");
    }
    std::cout << "\33[2K\r";
    epoch_loss = epoch_loss / mnist.size();

    std::cout << "Epoch " << epoch << ": Loss = " << epoch_loss << std::endl;
  }
  model.save("../../trained_models/mnist_float_100.hpp", "mnist_model");

  evaluate<float>(model);
  nn::quantization::quantize_dynamic(model);
  evaluate<float>(model);
}
