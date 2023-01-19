#include "autograd/autograd.hpp"
#include <iostream>

int main() {

    Variable x(5.0);
    Variable y(3.0);
    Variable z(10.0);

    auto a = (x * y) + z;
    auto b = a * y * x;

    b.backward();

    std::cout << "Results:" << std::endl;
    std::cout << b.value() << std::endl;

    std::cout << "dx = " << x.grad() << std::endl;
    std::cout << "dy = " << y.grad() << std::endl;
    std::cout << "dz = " << z.grad() << std::endl;
    std::cout << "da = " << a.grad() << std::endl;
    std::cout << "db = " << b.grad() << std::endl;
}