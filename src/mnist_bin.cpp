#include <algorithm>
#include <iostream>
#include <nn/all.hpp>
#include <nn/dataset/mnist_dataset.hpp>

#include "../trained_models/mnist_float_100.hpp"

extern const std::uint8_t _binary_t50_images_idx3_ubyte_start[];
extern const std::uint8_t _binary_t50_images_idx3_ubyte_end[];

extern const std::uint8_t _binary_t50_labels_idx1_ubyte_start[];
extern const std::uint8_t _binary_t50_labels_idx1_ubyte_end[];

int main() {
  auto model = nn::Sequential<float>();
  model.add(nn::Linear<float>(28 * 28, 100));
  model.add(nn::Sigmoid<float>());
  model.add(nn::Linear<float>(100, 10));
  model.init(mnist_model::mnist_model, mnist_model::mnist_model_len);

  std::cout << "Loading dataset..." << std::endl;
  auto mnist =
      nn::MnistDataset<float>(const_cast<std::uint8_t*>(&_binary_t50_images_idx3_ubyte_start[0]),
                              const_cast<std::uint8_t*>(&_binary_t50_images_idx3_ubyte_end[0]),
                              const_cast<std::uint8_t*>(&_binary_t50_labels_idx1_ubyte_start[0]),
                              const_cast<std::uint8_t*>(&_binary_t50_labels_idx1_ubyte_end[0]), 1);

  std::cout << "Starting evaluation..." << std::endl;
  int correct = 0;
  for (auto&& [img, label] : mnist) {
    auto input = img.view({1, 28 * 28});

    auto out = model(input);

    auto output = tensor::argmax(out, 1);
    if (output.item() == label.item()) {
      ++correct;
    }
  }
  std::cout << "Accuracy: " << (float)correct / mnist.count() << std::endl;
}
