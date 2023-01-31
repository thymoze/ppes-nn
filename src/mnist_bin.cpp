#include <algorithm>
#include <iostream>
#include <nn/dataset/mnist_dataset.hpp>
#include <nn/modules/linear.hpp>
#include <nn/modules/sigmoid.hpp>
#include <nn/mse.hpp>
#include <nn/optim/sgd.hpp>
#include <nn/sequential.hpp>
#include <tuple>

extern const std::uint8_t _binary_train_images_idx3_ubyte_start[];
extern const std::uint8_t _binary_train_images_idx3_ubyte_end[];

extern const std::uint8_t _binary_train_labels_idx1_ubyte_start[];
extern const std::uint8_t _binary_train_labels_idx1_ubyte_end[];

int main() {
  auto mnist =
      nn::MnistDataset<double>(const_cast<std::uint8_t*>(&_binary_train_images_idx3_ubyte_start[0]),
                       const_cast<std::uint8_t*>(&_binary_train_images_idx3_ubyte_end[0]),
                       const_cast<std::uint8_t*>(&_binary_train_labels_idx1_ubyte_start[0]),
                       const_cast<std::uint8_t*>(&_binary_train_labels_idx1_ubyte_end[0]));

  nn::Matrix<double> img, label;

  for (int idx = 0; idx < 4; idx++) {
    std::tie(img, label) = mnist.get(idx);
    std::cout << label(0, 0) << std::endl;
    for (std::size_t i = 0; i < img.rows(); i++) {
      for (std::size_t j = 0; j < img.cols(); j++) {
        std::cout << (img(i, j) == 0 ? "." : "@");
      }
      std::cout << std::endl;
    }
  }
}
