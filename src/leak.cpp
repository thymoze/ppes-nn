#include <autograd/autograd.hpp>
#include <matrix/matrix.hpp>

using namespace nn;

int main() {
  auto lhs = Variable(Matrix<double>(3, 3, 4));  // <---
  auto lhs_ = lhs;

  auto rhs = Variable(Matrix<double>(3, 3, 2));  // <---

  auto res = nn::sum(lhs * rhs);
  res.backward();  // <---

  auto mat1 = Matrix<double>(3, 3, 2);
  auto& mat2 = lhs.value();

  mat2 = mat2 - mat1;  // <---
}
