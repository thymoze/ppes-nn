#include <autograd/autograd.hpp>
#include <matrix/matrix.hpp>
#include <nn/sequential.hpp>
#include <nn/modules/linear.hpp>
#include <nn/modules/relu.hpp>
#include <nn/modules/sigmoid.hpp>
#include <nn/mse.hpp>
#include <nn/optim/sgd.hpp>

#include <iostream>
#include <numeric>
#include <utility>
#include <string>

int main() {
    std::vector<std::pair<Variable<double>, Variable<double>>> data = {
        { Variable<double>(Matrix<double>{1, 2, { 0, 0 }}), Variable<double>(Matrix<double>{1, 1, { 0 }}) },
        { Variable<double>(Matrix<double>{1, 2, { 1, 1 }}), Variable<double>(Matrix<double>{1, 1, { 0 }}) },
        { Variable<double>(Matrix<double>{1, 2, { 1, 0 }}), Variable<double>(Matrix<double>{1, 1, { 1 }}) },
        { Variable<double>(Matrix<double>{1, 2, { 0, 1 }}), Variable<double>(Matrix<double>{1, 1, { 1 }}) },
    };

    auto model = nn::Sequential();
    model.add(nn::Linear(2, 3));
    model.add(nn::Sigmoid());
    model.add(nn::Linear(3, 1));

    auto optimizer = SGD(model.params(), 0.1);

    for (int epoch = 0; epoch < 1000; epoch++) {
        double epoch_loss = 0;
        for (auto& [input, target] : data) {
            auto output = model({ input })[0];

            auto loss = mse(output, target);

            optimizer.zero_grad();
            loss.backward();

            optimizer.step();

            epoch_loss += loss(0,0);
        }
        epoch_loss = epoch_loss / data.size();

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ": Loss = " << epoch_loss << std::endl;
        }
    }

    // while (1) {
    //     std::string input;
    //     std::cout << "1. Input: ";
    //     std::getline(std::cin, input);
    //     double first = std::stoi(input);

    //     std::cout << "2. Input: ";
    //     std::getline(std::cin, input);
    //     double second = std::stoi(input);

    //     auto in = Variable<double>(Matrix<double>{1, 2, { first, second }});
    //     auto out = model({ in })[0];

    //     std::cout << "Output: " << out(0,0) << std::endl;
    // }
}