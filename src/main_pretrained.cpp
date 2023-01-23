#include "../trained_models/XOR_Model.hpp"
#include <iostream>
#include <utility>
#include <string>

int main(void)
{
    auto model = XOR_Model::create();
    while (1)
    {
        std::string input;
        std::cout << "1. Input: ";
        std::getline(std::cin, input);
        double first = std::stoi(input);

        std::cout << "2. Input: ";
        std::getline(std::cin, input);
        double second = std::stoi(input);

        auto in = Variable<double>(Matrix<double>{1, 2, {first, second}});
        auto out = model({in})[0];

        std::cout << "Output: " << out(0, 0) << std::endl;
    }
}
