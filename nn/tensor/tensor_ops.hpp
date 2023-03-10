#pragma once

#include <math.h>

#include <cassert>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <tensor/tensor_util.hpp>

namespace tensor {

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

Shape broadcast_shapes_for_matmul(const Shape& lhs, const Shape& rhs) {
  auto l_shape =
      std::vector(lhs.begin(), lhs.end() - std::min(lhs.size(), static_cast<std::size_t>(2)));
  auto r_shape =
      std::vector(rhs.begin(), rhs.end() - std::min(rhs.size(), static_cast<std::size_t>(2)));
  auto shape = broadcast_shapes(l_shape, r_shape);
  shape.push_back(*(lhs.end() - 2));
  shape.push_back(*(rhs.end() - 1));
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

  using ReduceIndexTensor = std::function<void(const Tensor<T>&, std::size_t, Tensor<T>&)>;
  using ReduceIndexEl = std::function<std::pair<T, std::size_t>(T, T, std::size_t, std::size_t)>;
  virtual ReduceIndexTensor reduce_index(ReduceIndexEl fn, T start) = 0;

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
    auto out = tensor::zeros<T>(t.shape(), t.f());
    neg_map_out(t, out);
    return out;
  }

  void id_map_out(const Tensor<T>& t, Tensor<T>& out) const {
    return ops_->map(std::identity())(t, out);
  }
  [[nodiscard]] Tensor<T> id_map(const Tensor<T>& t) const {
    auto out = tensor::zeros<T>(t.shape(), t.f());
    id_map_out(t, out);
    return out;
  }

  void inv_map_out(const Tensor<T>& t, Tensor<T>& out) const {
    ops_->map([](auto x) { return 1 / x; })(t, out);
  }
  [[nodiscard]] Tensor<T> inv_map(const Tensor<T>& t) const {
    auto out = tensor::zeros<T>(t.shape(), t.f());
    inv_map_out(t, out);
    return out;
  }

  void relu_map_out(const Tensor<T>& t, Tensor<T>& out) const {
    ops_->map([](auto x) { return x > 0 ? x : 0; })(t, out);
  }
  [[nodiscard]] Tensor<T> relu_map(const Tensor<T>& t) const {
    auto out = tensor::zeros<T>(t.shape(), t.f());
    relu_map_out(t, out);
    return out;
  }

  void exp_map_out(const Tensor<T>& t, Tensor<T>& out) const {
    ops_->map([](auto x) { return std::exp(x); })(t, out);
  }
  [[nodiscard]] Tensor<T> exp_map(const Tensor<T>& t) const {
    auto out = tensor::zeros<T>(t.shape(), t.f());
    exp_map_out(t, out);
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
    auto out = tensor::zeros<T>(t.shape(), t.f());
    sigmoid_map_out(t, out);
    return out;
  }

  // zip
  void add_zip_out(const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>& out) const {
    return ops_->zip(std::plus<T>())(lhs, rhs, out);
  }
  [[nodiscard]] Tensor<T> add_zip(const Tensor<T>& lhs, const Tensor<T>& rhs) const {
    auto shape = broadcast_shapes(lhs.shape(), rhs.shape());
    auto out = tensor::zeros<T>(shape, lhs.f());
    add_zip_out(lhs, rhs, out);
    return out;
  }

  void mul_zip_out(const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>& out) const {
    return ops_->zip(std::multiplies<T>())(lhs, rhs, out);
  }
  [[nodiscard]] Tensor<T> mul_zip(const Tensor<T>& lhs, const Tensor<T>& rhs) const {
    auto shape = broadcast_shapes(lhs.shape(), rhs.shape());
    auto out = tensor::zeros<T>(shape, lhs.f());
    mul_zip_out(lhs, rhs, out);
    return out;
  }

  void eq_zip_out(const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>& out) const {
    return ops_->zip(std::equal_to<T>())(lhs, rhs, out);
  }
  [[nodiscard]] Tensor<T> eq_zip(const Tensor<T>& lhs, const Tensor<T>& rhs) const {
    auto shape = broadcast_shapes(lhs.shape(), rhs.shape());
    auto out = tensor::zeros<T>(shape, lhs.f());
    eq_zip_out(lhs, rhs, out);
    return out;
  }

  void is_close_zip_out(const Tensor<T>& lhs, const Tensor<T>& rhs, float abs_tol, float rel_tol,
                        Tensor<T>& out) const {
    return ops_->zip([abs_tol, rel_tol](auto l, auto r) {
      return std::abs(l - r) <= abs_tol + rel_tol * std::abs(r);
    })(lhs, rhs, out);
  }
  [[nodiscard]] Tensor<T> is_close_zip(const Tensor<T>& lhs, const Tensor<T>& rhs, float abs_tol,
                                       float rel_tol) const {
    auto shape = broadcast_shapes(lhs.shape(), rhs.shape());
    auto out = tensor::zeros<T>(shape, lhs.f());
    is_close_zip_out(lhs, rhs, abs_tol, rel_tol, out);
    return out;
  }

  void inv_back_zip_out(const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>& out) const {
    return ops_->zip([](auto x, auto d) { return -d / (x * x); })(lhs, rhs, out);
  }
  [[nodiscard]] Tensor<T> inv_back_zip(const Tensor<T>& lhs, const Tensor<T>& rhs) const {
    auto shape = broadcast_shapes(lhs.shape(), rhs.shape());
    auto out = tensor::zeros<T>(shape, lhs.f());
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
    auto out = tensor::zeros<T>(shape, t.f());
    add_reduce_out(t, dim, out);
    return out;
  }

  void abs_add_reduce_out(const Tensor<T>& t, std::size_t dim, Tensor<T>& out) const {
    return ops_->reduce([](auto l, auto r) { return std::abs(l) + std::abs(r); }, 0)(t, dim, out);
  }
  [[nodiscard]] Tensor<T> abs_add_reduce(const Tensor<T>& t, std::size_t dim) const {
    auto shape = t.shape();
    shape[dim] = 1;
    auto out = tensor::zeros<T>(shape, t.f());
    abs_add_reduce_out(t, dim, out);
    return out;
  }

  void mul_reduce_out(const Tensor<T>& t, std::size_t dim, Tensor<T>& out) const {
    return ops_->reduce(std::multiplies<T>(), 1)(t, dim, out);
  }
  [[nodiscard]] Tensor<T> mul_reduce(const Tensor<T>& t, std::size_t dim) const {
    auto shape = t.shape();
    shape[dim] = 1;
    auto out = tensor::zeros<T>(shape, t.f());
    mul_reduce_out(t, dim, out);
    return out;
  }

  void all_reduce_out(const Tensor<T>& t, std::size_t dim, Tensor<T>& out) const {
    return ops_->reduce(std::logical_and(), 1)(t, dim, out);
  }
  [[nodiscard]] Tensor<T> all_reduce(const Tensor<T>& t, std::size_t dim) const {
    auto shape = t.shape();
    shape[dim] = 1;
    auto out = tensor::zeros<T>(shape, t.f());
    all_reduce_out(t, dim, out);
    return out;
  }

  void min_reduce_out(const Tensor<T>& t, std::size_t dim, Tensor<T>& out) const {
    return ops_->reduce([](auto l, auto r) { return std::min(l, r); },
                        std::numeric_limits<T>::max())(t, dim, out);
  }
  [[nodiscard]] Tensor<T> min_reduce(const Tensor<T>& t, std::size_t dim) const {
    auto shape = t.shape();
    shape[dim] = 1;
    auto out = tensor::zeros<T>(shape, t.f());
    min_reduce_out(t, dim, out);
    return out;
  }

  void max_reduce_out(const Tensor<T>& t, std::size_t dim, Tensor<T>& out) const {
    return ops_->reduce([](auto l, auto r) { return std::max(l, r); },
                        std::numeric_limits<T>::lowest())(t, dim, out);
  }
  [[nodiscard]] Tensor<T> max_reduce(const Tensor<T>& t, std::size_t dim) const {
    auto shape = t.shape();
    shape[dim] = 1;
    auto out = tensor::zeros<T>(shape, t.f());
    max_reduce_out(t, dim, out);
    return out;
  }

  void argmax_reduce_out(const Tensor<T>& t, std::size_t dim, Tensor<T>& out) const {
    return ops_->reduce_index(
        [](auto l, auto r, auto l_idx, auto r_idx) {
          return l >= r ? std::pair{l, l_idx} : std::pair{r, r_idx};
        },
        std::numeric_limits<T>::lowest())(t, dim, out);
  }
  [[nodiscard]] Tensor<T> argmax_reduce(const Tensor<T>& t, std::size_t dim) const {
    auto shape = t.shape();
    shape[dim] = 1;
    auto out = tensor::zeros<T>(shape, t.f());
    argmax_reduce_out(t, dim, out);
    return out;
  }

  void argmin_reduce_out(const Tensor<T>& t, std::size_t dim, Tensor<T>& out) const {
    return ops_->reduce_index(
        [](auto l, auto r, auto l_idx, auto r_idx) {
          return l <= r ? std::pair{l, l_idx} : std::pair{r, r_idx};
        },
        std::numeric_limits<T>::max())(t, dim, out);
  }
  [[nodiscard]] Tensor<T> argmin_reduce(const Tensor<T>& t, std::size_t dim) const {
    auto shape = t.shape();
    shape[dim] = 1;
    auto out = tensor::zeros<T>(shape, t.f());
    argmin_reduce_out(t, dim, out);
    return out;
  }

  [[nodiscard]] Tensor<T> matrix_multiply(const Tensor<T>& a, const Tensor<T>& b) const {
    if (*(a.shape().end() - 1) != *(b.shape().end() - 2)) {
      throw std::logic_error("Matrix multiplication dimensions don't match: " +
                             to_string(a.shape()) + " and " + to_string(b.shape()));
    }

    Tensor<T> lhs = a.ndims() == 2 ? a.unsqueeze(0) : a;
    Tensor<T> rhs = b.ndims() == 2 ? b.unsqueeze(0) : b;
    bool both_2d = a.ndims() == 2 && b.ndims() == 2;

    auto shape = broadcast_shapes_for_matmul(lhs.shape(), rhs.shape());

    auto out = tensor::zeros<T>(shape, a.f());
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
  typename TensorOps<T>::ZipTensor zip(typename TensorOps<T>::ZipEl fn) override {
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

  typename TensorOps<T>::MapTensor map(typename TensorOps<T>::MapEl fn) override {
    auto map_fn = [fn](const Tensor<T>& t, Tensor<T>& out) {
      Indices t_idx;
      for (auto& idx : out.indices()) {
        broadcasted_to_index_in_shape(idx, t.shape(), t_idx);

        out[idx] = fn(t[t_idx]);
      }
    };

    return map_fn;
  }

  typename TensorOps<T>::ReduceTensor reduce(typename TensorOps<T>::ReduceEl fn, T start) override {
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

  typename TensorOps<T>::ReduceIndexTensor reduce_index(typename TensorOps<T>::ReduceIndexEl fn,
                                                        T start) override {
    auto reduce_fn = [fn, start](const Tensor<T>& t, std::size_t dim, Tensor<T>& out) {
      auto dim_size = t.shape()[dim];
      for (auto& idx : out.indices()) {
        auto t_idx = idx;

        auto v = start;
        std::size_t ii = 0;
        for (std::size_t i = 0; i < dim_size; i++) {
          t_idx[dim] = i;
          std::tie(v, ii) = fn(v, t[t_idx], ii, i);
        }

        out[idx] = static_cast<T>(ii);
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

template <typename T>
class MTOps : public SimpleOps<T> {
 public:
  void matrix_multiply(const Tensor<T>& lhs, const Tensor<T>& rhs, Tensor<T>& out) override {
    auto indices = out.indices();
    std::vector<std::future<void>> futures;
    for (std::size_t b = 0; b <= out.size() / 500; b++) {
      futures.push_back(std::async([&out, &lhs, &rhs, &indices, &futures, b]() mutable {
        Indices lhs_idx, rhs_idx;
        auto block_end = std::min(indices.begin() + ((b + 1) * 500), indices.end());
        for (auto idx = indices.begin() + (b * 500); idx != block_end; ++idx) {
          auto out_idx = *idx;
          T v = 0;
          for (std::size_t k = 0; k < lhs.shape().back(); k++) {
            broadcasted_to_index_in_shape(out_idx, lhs.shape(), lhs_idx);
            *(lhs_idx.end() - 1) = k;

            broadcasted_to_index_in_shape(out_idx, rhs.shape(), rhs_idx);
            *(rhs_idx.end() - 2) = k;

            v += lhs[lhs_idx] * rhs[rhs_idx];
          }
          out[out_idx] = v;
        }
      }));
    }
  }
};

}  // namespace tensor
