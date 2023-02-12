#include <iostream>
#include <nn/all.hpp>

using tensor::Tensor;

int main() {
  nn::random::seed(0x5EED);

  std::pair<Tensor<double>, Tensor<double>> xor_data = {
      tensor::make<double>({4, 1, 2}, {0, 0, 1, 1, 1, 0, 0, 1}),
      tensor::make<double>({4, 1, 1}, {0, 0, 1, 1}),
  };

  auto model = nn::Sequential<double>();
  model.add(nn::Linear<double>(2, 3));
  model.add(nn::Sigmoid<double>());
  model.add(nn::Linear<double>(3, 1));
  model.init();

  auto optimizer = nn::SGD<double>(model.params(), 0.25);

  for (int epoch = 0; epoch < 2000; epoch++) {
    auto& [input, target] = xor_data;
    auto output = model(input);

    auto loss = nn::mse(output, target);

    optimizer.zero_grad();
    loss.backward();

    optimizer.step();

    if (epoch % 200 == 0) {
      std::cout << "\33[2K\r Epoch " << epoch << ": Loss = " << loss.item() << std::endl;
    }
  }

  model.save("../../trained_models/xor.hpp", "xor_model");

  // for (auto& param : model.params()) {
  //   std::cout << param.value() << std::endl;
  // }
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
