#pragma once

#include <math.h>

#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <tensor/tensor_util.hpp>

Shape broadcast_shapes(const Shape& lhs, const Shape& rhs) {
  Shape shape;
  auto size = std::max(lhs.size(), rhs.size());
  for (int i = 1; i <= static_cast<int>(size); i++) {
    auto l = (static_cast<int>(lhs.size()) - i < 0) ? 1 : lhs[lhs.size() - i];
    auto r = (static_cast<int>(rhs.size()) - i < 0) ? 1 : rhs[rhs.size() - i];

    if (l == r || l == 1 || r == 1) {
      shape.push_back(std::max(l, r));
    } else {
      throw std::logic_error("Shapes not broadcastable " + std::to_string(l) +
                             " != " + std::to_string(r) + " of " + to_string(lhs) + " and " +
                             to_string(rhs));
    }
  }
  std::reverse(shape.begin(), shape.end());
  return shape;
}

template <typename T>
class TensorOps {
 public:
  virtual ~TensorOps() = default;

  using MapTensor = std::function<void(const Tensor<T>&, Tensor<T>&)>;
  using MapEl = std::function<T(T)>;
  virtual MapTensor map(MapEl fn) = 0;

  using ZipTensor = std::function<void(const Tensor<T>&, const Tensor<T>&, Tensor<T>&)>;
  using ZipEl = std::function<T(T, T)>;
  virtual ZipTensor zip(ZipEl fn) = 0;

  using ReduceTensor = std::function<void(const Tensor<T>&, std::size_t, Tensor<T>&)>;
  using ReduceEl = std::function<T(T, T)>;
  virtual ReduceTensor reduce(ReduceEl fn, T start) = 0;

  virtual void matrix_multiply(const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>& out) = 0;
};

template <typename T>
class TensorBackend {
 public:
  template <typename O>
  explicit TensorBackend(O ops) : ops_(std::make_shared<O>(ops)) {}

  // map
  void neg_map_out(const Tensor<T>& t, Tensor<T>& out) const { ops_->map(std::negate())(t, out); }
  [[nodiscard]] Tensor<T> neg_map(const Tensor<T>& t) const {
    auto out = Tensor<T>::zeros(t.shape(), t.f());
    neg_map_out(t, out);
    return out;
  }

  void id_map_out(const Tensor<T>& t, Tensor<T>& out) const {
    return ops_->map(std::identity())(t, out);
  }
  [[nodiscard]] Tensor<T> id_map(const Tensor<T>& t) const {
    auto out = Tensor<T>::zeros(t.shape(), t.f());
    id_map_out(t, out);
    return out;
  }

  void inv_map_out(const Tensor<T>& t, Tensor<T>& out) const {
    ops_->map([](auto x) { return 1 / x; })(t, out);
  }
  [[nodiscard]] Tensor<T> inv_map(const Tensor<T>& t) const {
    auto out = Tensor<T>::zeros(t.shape(), t.f());
    inv_map_out(t, out);
    return out;
  }

  void relu_map_out(const Tensor<T>& t, Tensor<T>& out) const {
    ops_->map([](auto x) { return x > 0 ? x : 0; })(t, out);
  }
  [[nodiscard]] Tensor<T> relu_map(const Tensor<T>& t) const {
    auto out = Tensor<T>::zeros(t.shape(), t.f());
    relu_map_out(t, out);
    return out;
  }

  void sigmoid_map_out(const Tensor<T>& t, Tensor<T>& out) const {
    ops_->map([](auto x) {
      // This apparently has better numerical stability
      if (x >= 0) {
        return 1 / (1 + std::exp(-x));
      } else {
        auto e = std::exp(x);
        return e / (1 + e);
      }
    })(t, out);
  }
  [[nodiscard]] Tensor<T> sigmoid_map(const Tensor<T>& t) const {
    auto out = Tensor<T>::zeros(t.shape(), t.f());
    sigmoid_map_out(t, out);
    return out;
  }

  // zip
  void add_zip_out(const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>& out) const {
    return ops_->zip(std::plus<T>())(lhs, rhs, out);
  }
  [[nodiscard]] Tensor<T> add_zip(const Tensor<T>& lhs, const Tensor<T>& rhs) const {
    auto shape = broadcast_shapes(lhs.shape(), rhs.shape());
    auto out = Tensor<T>::zeros(shape, lhs.f());
    add_zip_out(lhs, rhs, out);
    return out;
  }

  void mul_zip_out(const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>& out) const {
    return ops_->zip(std::multiplies<T>())(lhs, rhs, out);
  }
  [[nodiscard]] Tensor<T> mul_zip(const Tensor<T>& lhs, const Tensor<T>& rhs) const {
    auto shape = broadcast_shapes(lhs.shape(), rhs.shape());
    auto out = Tensor<T>::zeros(shape, lhs.f());
    mul_zip_out(lhs, rhs, out);
    return out;
  }

  void inv_back_zip_out(const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>& out) const {
    return ops_->zip([](auto x, auto d) { return -d / (x * x); })(lhs, rhs, out);
  }
  [[nodiscard]] Tensor<T> inv_back_zip(const Tensor<T>& lhs, const Tensor<T>& rhs) const {
    auto shape = broadcast_shapes(lhs.shape(), rhs.shape());
    auto out = Tensor<T>::zeros(shape, lhs.f());
    inv_back_zip_out(lhs, rhs, out);
    return out;
  }

  // reduce
  void add_reduce_out(const Tensor<T>& t, std::size_t dim, Tensor<T>& out) const {
    return ops_->reduce(std::plus<T>(), 0)(t, dim, out);
  }
  [[nodiscard]] Tensor<T> add_reduce(const Tensor<T>& t, std::size_t dim) const {
    auto shape = t.shape();
    shape[dim] = 1;
    auto out = Tensor<T>::zeros(shape, t.f());
    add_reduce_out(t, dim, out);
    return out;
  }

  void mul_reduce_out(const Tensor<T>& t, std::size_t dim, Tensor<T>& out) const {
    return ops_->reduce(std::multiplies<T>(), 1)(t, dim, out);
  }
  [[nodiscard]] Tensor<T> mul_reduce(const Tensor<T>& t, std::size_t dim) const {
    auto shape = t.shape();
    shape[dim] = 1;
    auto out = Tensor<T>::zeros(shape, t.f());
    mul_reduce_out(t, dim, out);
    return out;
  }

  [[nodiscard]] Tensor<T> matrix_multiply(const Tensor<T>& a, const Tensor<T>& b) const {
    assert(*(a.shape().end() - 1) == *(b.shape().end() - 2) &&
           "Matrix multiplication dimensions don't match.");

    Tensor<T> lhs = a.ndims() == 2 ? a.reshape({1, a.shape()[0], a.shape()[1]}) : a;
    Tensor<T> rhs = b.ndims() == 2 ? b.reshape({1, b.shape()[0], b.shape()[1]}) : b;
    bool both_2d = a.ndims() == 2 && b.ndims() == 2;

    auto l_shape =
        std::vector(lhs.shape().begin(),
                    lhs.shape().end() - std::min(lhs.shape().size(), static_cast<std::size_t>(2)));
    auto r_shape =
        std::vector(rhs.shape().begin(),
                    rhs.shape().end() - std::min(rhs.shape().size(), static_cast<std::size_t>(2)));
    auto shape = broadcast_shapes(l_shape, r_shape);
    shape.push_back(*(a.shape().end() - 2));
    shape.push_back(*(b.shape().end() - 1));

    auto out = Tensor<T>::zeros(shape, a.f());
    ops_->matrix_multiply(lhs, rhs, out);

    if (both_2d) {
      out = out.view({*(shape.end() - 2), *(shape.end() - 1)});
    }

    return out;
  }

 private:
  std::shared_ptr<TensorOps<T>> ops_;
};

template <typename T>
class SimpleOps : public TensorOps<T> {
 public:
  TensorOps<T>::ZipTensor zip(TensorOps<T>::ZipEl fn) override {
    auto zip_fn = [fn](const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>& out) {
      Indices lhs_idx, rhs_idx;
      for (auto& idx : out.indices()) {
        broadcasted_to_index_in_shape(idx, lhs.shape(), lhs_idx);
        broadcasted_to_index_in_shape(idx, rhs.shape(), rhs_idx);

        out[idx] = fn(lhs[lhs_idx], rhs[rhs_idx]);
      }
    };

    return zip_fn;
  }

  TensorOps<T>::MapTensor map(TensorOps<T>::MapEl fn) override {
    auto map_fn = [fn](const Tensor<T>& t, Tensor<T>& out) {
      if (std::equal(t.strides().begin(), t.strides().end(), out.strides().begin(),
                     out.strides().end())) {
        std::transform(t.data().data()->begin(), t.data().data()->end(), out.data().data()->begin(),
                       fn);
      } else {
        Indices t_idx;
        for (auto& idx : out.indices()) {
          broadcasted_to_index_in_shape(idx, t.shape(), t_idx);

          out[idx] = fn(t[t_idx]);
        }
      }
    };

    return map_fn;
  }

  TensorOps<T>::ReduceTensor reduce(TensorOps<T>::ReduceEl fn, T start) override {
    auto reduce_fn = [fn, start](const Tensor<T>& t, std::size_t dim, Tensor<T>& out) {
      auto dim_size = t.shape()[dim];
      for (auto& idx : out.indices()) {
        auto t_idx = idx;

        auto v = start;
        for (std::size_t i = 0; i < dim_size; i++) {
          t_idx[dim] = i;
          v = fn(v, t[t_idx]);
        }

        out[idx] = v;
      }
    };

    return reduce_fn;
  }

  void matrix_multiply(const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>& out) override {
    Indices lhs_idx, rhs_idx;
    for (auto& idx : out.indices()) {
      T v = 0;
      for (std::size_t k = 0; k < lhs.shape().back(); k++) {
        broadcasted_to_index_in_shape(idx, lhs.shape(), lhs_idx);
        *(lhs_idx.end() - 1) = k;

        broadcasted_to_index_in_shape(idx, rhs.shape(), rhs_idx);
        *(rhs_idx.end() - 2) = k;

        v += lhs[lhs_idx] * rhs[rhs_idx];
      }
      out[idx] = v;
    }
  }
};
