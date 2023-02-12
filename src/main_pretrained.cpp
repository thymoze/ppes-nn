#include <iostream>
#include <nn/dataset/mnist_dataset.hpp>
#include <string>
#include <utility>

#include "../trained_models/MNIST.hpp"
#include "../trained_models/XOR_Model.hpp"

// int main(void) {
//   auto model = XOR_Model<float>::create();
//   while (1) {
//     std::string input;
//     std::cout << "1. Input: ";
//     std::getline(std::cin, input);
//     float first = std::stoi(input);

//     std::cout << "2. Input: ";
//     std::getline(std::cin, input);
//     float second = std::stoi(input);

//     auto in = nn::Variable<float>(nn::Matrix<float>{1, 2, {first, second}});
//     auto out = model({in})[0];

//     std::cout << "Output: " << out(0, 0) << std::endl;
//   }
// }

int main() {
  auto mnist = nn::MnistDataset<float>("../../data", nn::MnistDataset<float>::Set::TEST);
  auto model = MNIST<float>::create();

  int correct = 0;

  for (auto &[input, target] : mnist) {
    input.reshape(1, 28 * 28);
    auto output_onehot = model({input})[0];

    auto target_onehot = nn::Matrix<float>(1, 10, 0);
    target_onehot(0, static_cast<int>(target(0, 0))) = 1;

    auto output =
        std::distance(output_onehot.value().begin(),
                      std::max_element(output_onehot.value().begin(), output_onehot.value().end()));
    if (output == target(0, 0)) {
      correct++;
    }
  }

  std::cout << "accuracy: " << static_cast<double>(correct) / mnist.size() << std::endl;
  ;
}
