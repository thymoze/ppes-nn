#pragma once

#include <autograd/variable.hpp>
#include <matrix/matrix.hpp>
#include <vector>

namespace nn {

template <typename T>
Variable<T> operator+(const Variable<T>& lhs, const Variable<T>& rhs) {
  auto result = lhs.value() + rhs.value();

  auto gradFunc = [](const std::vector<Variable<T>>& inputs, const Variable<T>& grad) {
    inputs[0].add_grad(grad);
    inputs[1].add_grad(grad);
  };

  return Variable<T>(result, {lhs, rhs}, gradFunc);
}

template <typename T>
Variable<T> operator-(const Variable<T>& lhs, const Variable<T>& rhs) {
  auto result = lhs.value() - rhs.value();

  auto gradFunc = [](const std::vector<Variable<T>>& inputs, const Variable<T>& grad) {
    inputs[0].add_grad(grad);
    inputs[1].add_grad(negate(grad));
  };

  return Variable<T>(result, {lhs, rhs}, gradFunc);
}

template <typename T>
Variable<T> operator*(const Variable<T>& lhs, double scalar) {
  auto result = lhs.value() * scalar;

  auto gradFunc = [scalar](const std::vector<Variable<T>>& inputs, const Variable<T>& grad) {
    inputs[0].add_grad(grad * scalar);
  };

  return Variable<T>{result, {lhs}, gradFunc};
}

template <typename T>
Variable<T> operator*(double scalar, const Variable<T>& rhs) {
  return rhs * scalar;
}

template <typename T>
Variable<T> operator*(const Variable<T>& lhs, const Variable<T>& rhs) {
  auto result = lhs.value() * rhs.value();

  auto gradFunc = [](const std::vector<Variable<T>>& inputs, const Variable<T>& grad) {
    inputs[0].add_grad(grad * inputs[1]);
    inputs[1].add_grad(grad * inputs[0]);
  };

  return Variable<T>{result, {lhs, rhs}, gradFunc};
}

template <typename T>
Variable<T> negate(const Variable<T>& var) {
  auto result = var.value() * -1;

  auto gradFunc = [](const std::vector<Variable<T>>& inputs, const Variable<T>& grad) {
    inputs[0].add_grad(negate(grad));
  };

  return Variable<T>{result, {var}, gradFunc};
}

template <typename T>
Variable<T> matmul(const Variable<T>& lhs, const Variable<T>& rhs) {
  auto result = lhs.value().matmul(rhs.value());

  auto gradFunc = [](const std::vector<Variable<T>>& inputs, const Variable<T>& grad) {
    inputs[0].add_grad(matmul(grad, inputs[1].transpose()));
    inputs[1].add_grad(matmul(inputs[0].transpose(), grad));
  };

  return Variable<T>{result, {lhs, rhs}, gradFunc};
}

template <typename T>
Variable<T> sum(const Variable<T>& var) {
  auto result = sum(var.value());

  auto gradFunc = [](const std::vector<Variable<T>>& inputs, const Variable<T>& grad) {
    inputs[0].add_grad(
        Variable<T>(Matrix<T>(inputs[0].rows(), inputs[0].cols(), grad.value()(0, 0))));
  };

  return Variable<T>{result, {var}, gradFunc};
}

template <typename T>
Variable<T> mean(const Variable<T>& var) {
  auto result = mean(var.value());

  auto gradFunc = [](const std::vector<Variable<T>>& inputs, const Variable<T>& grad) {
    T count = static_cast<T>(inputs[0].rows() * inputs[0].cols());
    inputs[0].add_grad(
        Variable<T>(Matrix<T>(inputs[0].rows(), inputs[0].cols(), grad.value()(0, 0) / count)));
  };

  return Variable<T>{result, {var}, gradFunc};
}

template <typename T>
Variable<T> max(const Variable<T>& var, T value) {
  auto result = max(var.value(), value);

  auto gradFunc = [value](const std::vector<Variable<T>>& inputs, const Variable<T>& grad) {
    auto g = Matrix<T>(inputs[0].rows(), inputs[0].cols());

    for (std::size_t i = 0; i < inputs[0].rows(); i++) {
      for (std::size_t j = 0; j < inputs[0].cols(); j++) {
        if (inputs[0](i, j) >= value) {
          g(i, j) = grad(i, j);
        } else {
          g(i, j) = 0;
        }
      }
    }

    inputs[0].add_grad(Variable<T>(g));
  };

  return Variable<T>{result, {var}, gradFunc};
}

template <typename T>
Variable<T> exp(const Variable<T>& var) {
  auto result = exp(var.value());

  auto gradFunc = [](const std::vector<Variable<T>>& inputs, const Variable<T>& grad) {
    inputs[0].add_grad(grad * exp(inputs[0]));
  };

  return Variable<T>{result, {var}, gradFunc};
}

template <typename T>
Variable<T> reciprocal(const Variable<T>& denominator) {
  auto ones = Matrix<double>(denominator.rows(), denominator.cols(), 1);
  auto result = ones / denominator.value();

  auto gradFunc = [](const std::vector<Variable<T>>& inputs, const Variable<T>& grad) {
    inputs[0].add_grad(grad * negate(reciprocal(inputs[0] * inputs[0])));
  };

  return Variable<T>{result, {denominator}, gradFunc};
}

}  // namespace nn
