#pragma once

#include <functional>
#include <matrix/matrix.hpp>
#include <memory>
#include <unordered_set>
#include <vector>

/// A container around data to facilitate reverse-mode automatic differentiation
template <typename T>
class Variable {
 public:
  using GradFunc = std::function<void(const std::vector<Variable>&, const Variable<T>&)>;

  Variable() = default;

  Variable(Matrix<T> value) : shared_value(std::make_shared<Matrix<T>>(std::move(value))) {}

  Variable(Matrix<T> value, std::vector<Variable> inputs, GradFunc gradFunc)
      : shared_value(std::make_shared<Matrix<T>>(std::move(value))) {
    shared_grad->inputs = std::move(inputs);
    shared_grad->gradFunc = std::move(gradFunc);
  }

  void backward() const {
    add_grad(Matrix<T>(1, 1, {1}));

    auto dag = build();
    for (auto iter = dag.rbegin(); iter != dag.rend(); iter++) {
      if (iter->shared_grad->gradFunc) {
        iter->shared_grad->gradFunc(iter->shared_grad->inputs, *iter->shared_grad->grad);
      }
    }
  }

  void add_grad(const Variable<T>& grad) const {
    if (shared_grad->grad) {
      shared_grad->grad =
          std::make_unique<Variable<T>>(Variable(shared_grad->grad->value() + grad.value()));
    } else {
      shared_grad->grad = std::make_unique<Variable<T>>(grad);
    }
  }

  void zero_grad() const { shared_grad->grad.reset(); }

  Variable<T>& grad() const { return *shared_grad->grad; }

  Matrix<T>& value() const { return *shared_value; }

 private:
  std::vector<Variable> build() const {
    std::unordered_set<SharedGrad*> cache;
    std::vector<Variable> dag;
    std::function<void(const Variable&)> recurse;

    // Topological sort
    recurse = [&](const Variable& var) {
      auto id = var.shared_grad.get();
      if (cache.find(id) != cache.end()) {
        return;
      }
      for (const auto& input : var.shared_grad->inputs) {
        recurse(input);
      }
      cache.insert(id);
      dag.push_back(var);
    };

    recurse(*this);
    return dag;
  }

  struct SharedGrad {
    std::unique_ptr<Variable<T>> grad{nullptr};
    std::vector<Variable> inputs;
    GradFunc gradFunc{nullptr};
  };

  std::shared_ptr<SharedGrad> shared_grad = std::make_shared<SharedGrad>();
  std::shared_ptr<Matrix<T>> shared_value = std::make_shared<Matrix<T>>();

 public:
  // Functions passed through to Matrix

  std::size_t rows() const { return value().rows(); }

  std::size_t cols() const { return value().cols(); }

  std::pair<std::size_t, std::size_t> size() const { return value().size(); }

  Matrix<T>::ref operator()(size_t i, size_t j) { return value().operator()(i, j); }

  Matrix<T>::const_ref operator()(size_t i, size_t j) const { return value().operator()(i, j); }

  Variable<T> transpose() const {
    return Variable(value().transpose(), shared_grad->inputs, shared_grad->gradFunc);
  }
};
