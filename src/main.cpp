#include <autograd/autograd.hpp>
#include <filesystem>
#include <iostream>
#include <matrix/matrix.hpp>
#include <nn/modules/linear.hpp>
#include <nn/modules/relu.hpp>
#include <nn/modules/sigmoid.hpp>
#include <nn/mse.hpp>
#include <nn/optim/sgd.hpp>
#include <nn/sequential.hpp>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>

using var = nn::Variable<float>;

int main() {
  auto test = std::filesystem::current_path();
  std::cout << test.string() << std::endl;
}

// int main() {
//   std::vector<std::pair<var, var>> xor_data = {
//       {var(nn::Matrix<float>{1, 2, {0, 0}}), var(nn::Matrix<float>{1, 1, {0}})},
//       {var(nn::Matrix<float>{1, 2, {1, 1}}), var(nn::Matrix<float>{1, 1, {0}})},
//       {var(nn::Matrix<float>{1, 2, {1, 0}}), var(nn::Matrix<float>{1, 1, {1}})},
//       {var(nn::Matrix<float>{1, 2, {0, 1}}), var(nn::Matrix<float>{1, 1, {1}})},
//   };

//   std::vector<std::pair<var, var>> and_data = {
//       {var(nn::Matrix<float>{1, 2, {0, 0}}), var(nn::Matrix<float>{1, 1, {0}})},
//       {var(nn::Matrix<float>{1, 2, {1, 1}}), var(nn::Matrix<float>{1, 1, {1}})},
//       {var(nn::Matrix<float>{1, 2, {1, 0}}), var(nn::Matrix<float>{1, 1, {0}})},
//       {var(nn::Matrix<float>{1, 2, {0, 1}}), var(nn::Matrix<float>{1, 1, {0}})},
//   };

//   auto model = nn::Sequential<float>();
//   model.add(nn::Linear<float>(2, 3));
//   model.add(nn::Sigmoid<float>());
//   model.add(nn::Linear<float>(3, 1));

//   auto optimizer = nn::SGD(model.params(), 0.1);

//   for (int epoch = 0; epoch < 1000; epoch++) {
//     double epoch_loss = 0;
//     for (auto &[input, target] : xor_data) {
//       auto output = model({input})[0];

//       auto loss = nn::mse(output, target);

//       optimizer.zero_grad();
//       loss.backward();
//       optimizer.step();
//       loss.reset_dag();

//       epoch_loss += loss(0, 0);
//     }
//     epoch_loss = epoch_loss / and_data.size();

//     if (epoch % 100 == 0) {
//       std::cout << "Epoch " << epoch << ": Loss = " << epoch_loss << std::endl;
//     }
//   }

//   std::cout << "hello" << std::endl;

//   // model.save("XOR_Model");
// }
