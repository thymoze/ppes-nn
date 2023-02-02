#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <tensor/tensor_util.hpp>
#include <unordered_map>
#include <unordered_set>

template <typename T>
class Context {
 public:
  explicit Context(bool no_grad) : no_grad_(no_grad), saved_values_() {}

  void save_for_backward(std::vector<Tensor<T>> values) {
    if (no_grad_) {
      return;
    }
    saved_values_.reserve(saved_values_.size() + values.size());
    std::move(values.begin(), values.end(), std::back_inserter(saved_values_));
  }

  [[nodiscard]] const std::vector<Tensor<T>>& saved_values() const { return saved_values_; }

 private:
  bool no_grad_;
  std::vector<Tensor<T>> saved_values_;
};

template <typename T>
struct History {
  std::unique_ptr<Function<T>> last_fn;
  std::unique_ptr<Context<T>> ctx;
  std::vector<Tensor<T>> inputs;
};

template <typename T>
std::vector<Tensor<T>> topological_sort(const Tensor<T>& root) {
  std::unordered_set<int> cache;
  std::vector<Tensor<T>> dag;
  std::function<void(const Tensor<T>&)> recurse;

  recurse = [&](const Tensor<T>& t) {
    if (t.is_constant()) return;

    auto id = t.id();
    if (cache.find(id) != cache.end()) {
      return;
    }
    for (const auto& input : t.parents()) {
      recurse(input);
    }
    cache.insert(id);
    dag.push_back(t);
  };
  recurse(root);

  return dag;
}

template <typename T>
void backpropagate(const Tensor<T>& variable, const Tensor<T>& grad) {
  auto dag = topological_sort(variable);

  std::unordered_map<std::size_t, Tensor<T>> derivatives;
  derivatives.insert({variable.id(), grad});
  for (auto iter = dag.rbegin(); iter != dag.rend(); iter++) {
    auto& t = *iter;
    assert(derivatives.find(t.id()) != derivatives.end() && "Must have a derivative.");
    if (t.is_leaf()) {
      t.add_grad(derivatives.at(t.id()));
    } else {
      auto gradients = t.backprop_step(derivatives.at(t.id()));
      for (auto& [input, g] : gradients) {
        auto id = input.id();
        if (derivatives.find(id) != derivatives.end()) {
          derivatives.at(id) = derivatives.at(id) + g;
        } else {
          derivatives.insert({id, g});
        }
      }
    }
  }
}
