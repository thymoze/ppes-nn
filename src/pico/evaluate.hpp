#include <pico/stdlib.h>

#include <cstdint>
#include <nn/all.hpp>
#include <nn/dataset/mnist_dataset.hpp>

extern const std::uint8_t _binary_t50_images_idx3_ubyte_start[];
extern const std::uint8_t _binary_t50_images_idx3_ubyte_end[];

extern const std::uint8_t _binary_t50_labels_idx1_ubyte_start[];
extern const std::uint8_t _binary_t50_labels_idx1_ubyte_end[];

template <typename T>
void evaluate(nn::Module<T>* model) {
  std::cout << "Loading dataset..." << std::endl;
  auto mnist =
      nn::MnistDataset<T>(const_cast<std::uint8_t*>(&_binary_t50_images_idx3_ubyte_start[0]),
                          const_cast<std::uint8_t*>(&_binary_t50_images_idx3_ubyte_end[0]),
                          const_cast<std::uint8_t*>(&_binary_t50_labels_idx1_ubyte_start[0]),
                          const_cast<std::uint8_t*>(&_binary_t50_labels_idx1_ubyte_end[0]), 1);

  std::cout << "Starting evaluation..." << std::endl;
  auto start = get_absolute_time();
  int correct = 0;
  for (auto&& [img, label] : mnist) {
    auto input = img.view({1, 28 * 28});

    auto out = (*model)(input);

    auto output = tensor::argmax(out, 1);
    if (output.item() == label.item()) {
      ++correct;
    }
  }
  auto end = get_absolute_time();
  std::cout << "Finished in " << end - start << " us" << std::endl;
  std::cout << "Accuracy: " << (float)correct / mnist.count() << std::endl;
}
