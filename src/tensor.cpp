#include <vector>
#include <tensor/tensor.hpp>
#include <iostream>
#include <algorithm>

int main() {
  std::vector<double> data(6);
  std::iota(data.begin(), data.end(), 0);

  auto s = Tensor<double>::make({2, 3}, std::move(data));
  s.requires_grad(true);
  auto t = Tensor<double>::make({2, 3}, {3, 3, 3, 3, 3, 3});
  t.requires_grad(true);

  auto u = (s + t) * (s + t);
  auto r = mean(u);

  r.backward();

  std::cout << s << std::endl;
  std::cout << t << std::endl;
  std::cout << r << std::endl;
  std::cout << std::endl;
  std::cout << (static_cast<bool>(s.grad()) ? (*s.grad()).to_string() : "s has no grad")
            << std::endl;
  std::cout << (static_cast<bool>(t.grad()) ? (*t.grad()).to_string() : "t has no grad")
            << std::endl;
}
