#pragma once

#include <algorithm>
#include <nn/random.hpp>
#include <optional>
#include <tensor/autodiff.hpp>
#include <tensor/tensor_data.hpp>
#include <tensor/tensor_functions.hpp>
#include <tensor/tensor_ops.hpp>
#include <tensor/tensor_util.hpp>

namespace tensor {

template <typename T>
class Tensor {
 public:
  Tensor(const Tensor<T>& other)
      : data_(other.data_->clone()),
        backend_(other.backend_),
        history_(other.history_),
        grad_(other.grad_) {}

  Tensor<T>& operator=(const Tensor<T>& other) {
    if (this != &other) {
      data_ = other.data_->clone();
      backend_ = other.backend_;
      history_ = other.history_;
      grad_ = other.grad_;
    }
    return *this;
  }

  explicit Tensor() = default;
  explicit Tensor(std::unique_ptr<TensorData<T>> data)
      : Tensor(std::move(data), DEFAULT_TENSOR_BACKEND) {}
  explicit Tensor(std::unique_ptr<TensorData<T>> data, TensorBackend<T> backend)
      : data_(std::move(data)), backend_(std::move(backend)) {}

  explicit Tensor(Tensor<T>&& t, History<T>&& history)
      : data_(std::move(t.data_)),
        backend_(std::move(t.backend_)),
        history_(std::make_shared<History<T>>(std::move(history))) {}

  [[nodiscard]] bool requires_grad() const { return static_cast<bool>(history_); }

  void requires_grad(bool grad) {
    if (grad) {
      history_ = std::make_shared<History<T>>();
    } else {
      history_.reset();
    }
  }

  [[nodiscard]] Tensor<T> detach() const { return Tensor<T>(data_->clone(), backend_); }

  [[nodiscard]] Tensor<T> expand(const Tensor<T>& other) const;

  void backward();
  void add_grad(const Tensor<T>& x);
  void zero_grad() { *grad_ = std::nullopt; }

  /// True, if the tensor was created by the user, i.e. does not have a `last_fn`
  [[nodiscard]] bool is_leaf() const { return history_ && !history_->last_fn; }
  [[nodiscard]] bool is_constant() const { return !history_; }
  [[nodiscard]] std::vector<Tensor<T>> parents() const {
    return (history_ ? history_->inputs : std::vector<Tensor<T>>());
  }
  [[nodiscard]] std::size_t id() const { return reinterpret_cast<std::size_t>(history_.get()); }
  [[nodiscard]] std::vector<std::pair<Tensor<T>, Tensor<T>>> backprop_step(
      const Tensor<T>& grad) const;

  [[nodiscard]] const std::optional<Tensor<T>>& grad() { return *grad_; }

  [[nodiscard]] const TensorBackend<T>& f() const { return backend_; }

  [[nodiscard]] typename TensorData<T>::const_ref item() const {
    assert(data_->size() == 1 && "item() only works on single-item tensors.");
    return data_->at(0);
  }

  template <typename... Idx>
  typename TensorData<T>::ref operator()(Idx... idx) {
    return data_->get({static_cast<std::size_t>(idx)...});
  }

  template <typename... Idx>
  typename TensorData<T>::const_ref operator()(Idx... idx) const {
    return data_->get({static_cast<std::size_t>(idx)...});
  }

  typename TensorData<T>::ref operator[](const Indices& idx) { return data_->get(idx); }
  typename TensorData<T>::const_ref operator[](const Indices& idx) const { return data_->get(idx); }

  [[nodiscard]] Tensor<T> contiguous() const { return Copy<T>()(*this); }
  [[nodiscard]] Tensor<T> view(const Shape& shape) const {
    std::vector<T> new_shape(shape.size());
    std::transform(shape.begin(), shape.end(), new_shape.begin(),
                   [](auto v) { return static_cast<T>(v); });
    return View<T>()(*this, tensor::make<T>(std::move(new_shape)));
  }
  [[nodiscard]] Tensor<T> reshape(const Shape& shape) const {
    if (data_.is_contiguous()) {
      return view(shape);
    } else {
      return contiguous().view(shape);
    }
  }
  [[nodiscard]] Tensor<T> squeeze() const { return Squeeze<T>()(*this); };
  [[nodiscard]] Tensor<T> unsqueeze(std::size_t dim) const {
    return Unsqueeze<T>()(*this, tensor::make<T>(static_cast<T>(dim)));
  };

  [[nodiscard]] const TensorData<T>* data() const { return data_.get(); }

  [[nodiscard]] IndicesIterator indices() const { return data_->indices(); }

  [[nodiscard]] const Shape& shape() const { return data_->shape(); }
  [[nodiscard]] const Strides& strides() const { return data_->strides(); }
  [[nodiscard]] std::size_t size() const { return data_->size(); }
  [[nodiscard]] std::size_t ndims() const { return data_->ndims(); }
  [[nodiscard]] std::string to_string() const { return data_->to_string(); }

  Tensor<T> to(TensorBackend<T> backend) {
    backend_ = std::move(backend);
    return *this;
  }

 protected:
  std::unique_ptr<TensorData<T>> data_;
  TensorBackend<T> backend_ = DEFAULT_TENSOR_BACKEND;
  std::shared_ptr<History<T>> history_ = std::shared_ptr<History<T>>();
  std::shared_ptr<std::optional<Tensor<T>>> grad_ =
      std::make_shared<std::optional<Tensor<T>>>(std::nullopt);
};

template <typename T>
void Tensor<T>::backward() {
  auto grad = tensor::make<T>({1}, {1}, backend_);
  backpropagate(*this, grad);
}

template <typename T>
void Tensor<T>::add_grad(const Tensor<T>& x) {
  assert(is_leaf() && "Only leaves should have derivatives.");
  if (!grad_->has_value()) {
    *grad_ = tensor::zeros<T>(data_->shape(), backend_);
  }
  *grad_ = grad_->value() + x;
}

template <typename T>
std::vector<std::pair<Tensor<T>, Tensor<T>>> Tensor<T>::backprop_step(const Tensor<T>& grad) const {
  assert(history_ && history_->last_fn && history_->ctx);

  auto y = history_->last_fn->backward(*history_->ctx, grad);
  assert(history_->inputs.size() == y.size() &&
         "Backward pass must produce a result for each input");

  std::vector<std::pair<Tensor<T>, Tensor<T>>> result;
  for (std::size_t i = 0; i < y.size(); i++) {
    result.emplace_back(std::make_pair(history_->inputs[i], history_->inputs[i].expand(y[i])));
  }
  return result;
};

template <typename T>
Tensor<T> Tensor<T>::expand(const Tensor<T>& other) const {
  // Case 1: Both the same shape
  if (ndims() == other.ndims() &&
      std::equal(shape().begin(), shape().end(), other.shape().begin())) {
    return other;
  }

  // Case 2: Backward is smaller, broadcast up
  auto broadcast_shape = broadcast_shapes(shape(), other.shape());
  auto buf = tensor::zeros<T>(broadcast_shape, backend_);
  backend_.id_map_out(other, buf);
  if (ndims() == buf.ndims() && std::equal(shape().begin(), shape().end(), buf.shape().begin())) {
    return buf;
  }

  // Case 3: Still different, reduce extra dims
  Shape orig_shape = shape();
  orig_shape.insert(orig_shape.begin(), buf.shape().size() - shape().size(), 1);
  for (std::size_t dim = 0; dim < broadcast_shape.size(); dim++) {
    if (orig_shape[dim] == 1 && broadcast_shape[dim] != 1) {
      buf = backend_.add_reduce(buf, dim);
    }
  }

  assert(data_->size() == buf.size() && "Expanding shapes failed");
  return buf;
}

template <typename T>
Tensor<T> operator-(const Tensor<T>& t) {
  return Neg<T>()(t);
}

template <typename T>
Tensor<T> operator*(const Tensor<T>& lhs, const Tensor<T>& rhs) {
  return Mul<T>()(lhs, rhs);
}

template <typename T>
Tensor<T> operator/(const Tensor<T>& lhs, const Tensor<T>& rhs) {
  return Mul<T>()(lhs, Inv<T>()(rhs));
}

template <typename T>
Tensor<T> operator+(const Tensor<T>& lhs, const Tensor<T>& rhs) {
  return Add<T>()(lhs, rhs);
}

template <typename T>
Tensor<T> operator-(const Tensor<T>& lhs, const Tensor<T>& rhs) {
  return Add<T>()(lhs, -rhs);
}

// Calculates the sum of each slice in the input along the given dimension. If no dimension is
// supplied, the sum of all elements is returned.
template <typename T>
Tensor<T> sum(const Tensor<T>& t, std::optional<std::size_t> dim = std::nullopt) {
  if (!dim) {
    return Sum<T>()(t.contiguous().view({t.size()}), tensor::make<T>(0));
  } else {
    return Sum<T>()(t, tensor::make<T>(*dim));
  }
}

// Calculates the mean of each slice in the input along the given dimension. If no dimension is
// supplied, the mean of all elements is returned.
template <typename T>
Tensor<T> mean(const Tensor<T>& t, std::optional<std::size_t> dim = std::nullopt) {
  if (!dim) {
    return sum(t, dim) / tensor::make<T>(t.size());
  } else {
    return sum(t, dim) / tensor::make<T>(t.shape()[*dim]);
  }
}

// Applies the ReLU function.
template <typename T>
Tensor<T> relu(const Tensor<T>& t) {
  return ReLU<T>()(t);
}

// Applies the sigmoid function.
template <typename T>
Tensor<T> sigmoid(const Tensor<T>& t) {
  return Sigmoid<T>()(t);
}

// Applies the softmax function along a given dimension (i.e. every slice along dim will sum to 1)
template <typename T>
Tensor<T> softmax(const Tensor<T>& t, std::size_t dim) {
  auto e = Exp<T>()(t);
  return e / sum(e, dim);
}

// Calculates the matrix-matrix product of two tensors. For tensors >2d a batched
// matrix-multiplication is performed.
template <typename T>
Tensor<T> matmul(const Tensor<T>& lhs, const Tensor<T>& rhs) {
  return MatMul<T>()(lhs, rhs);
}

// Non-differentiable operations:
// (because it was not implemented yet, not because it would not be possible :) )

template <typename T, typename It>
[[nodiscard]] Tensor<T> stack(It start, It end) {
  assert(std::distance(start, end) >= 1 && "Need at least 1 tensor to stack.");
  auto& backend = (*start).f();

  Shape shape = (*start).shape();
  shape.insert(shape.begin(), std::distance(start, end));

  Strides strides = (*start).strides();
  strides.insert(strides.begin(), (*start).size());

  auto data = std::vector<T>();
  data.reserve((*start).size() * std::distance(start, end));

  for (auto& t = start; t != end; ++t) {
    assert(std::equal(shape.begin() + 1, shape.end(), (*t).shape().begin(), (*t).shape().end()) &&
           "All stacked tensors need to be of equal shape.");
    assert(std::equal(strides.begin() + 1, strides.end(), (*t).strides().begin(),
                      (*t).strides().end()) &&
           "All stacked tensors need to have equal strides.");

    data.insert(data.end(), t->data()->begin(), t->data()->end());
  }

  auto tensor = std::make_unique<VectorStorage<T>>(
      std::make_shared<std::vector<T>>(std::move(data)), std::move(strides), std::move(shape));
  return Tensor<T>{std::move(tensor), backend};
}

// Element-wise equality comparison, returns a Tensor of 1s and 0s for equal and non-equal elements.
template <typename T>
Tensor<T> operator==(const Tensor<T>& lhs, const Tensor<T>& rhs) {
  return lhs.f().eq_zip(lhs, rhs);
}

// Element-wise "closeness" comparison, where closeness is defined as:
// > |lhs - rhs| <= abs_tol + rel_tol * |rhs|
// Returns a Tensor of 1s and 0s for close and non-close elements.
template <typename T>
Tensor<T> is_close(const Tensor<T>& lhs, const Tensor<T>& rhs, float abs_tol = 1e-8,
                   float rel_tol = 1e-5) {
  return lhs.f().is_close_zip(lhs, rhs, abs_tol, rel_tol);
}

// Tests if all elements in the input are truthy (optionally only along a given dimension).
template <typename T>
Tensor<T> all(const Tensor<T>& t, std::optional<std::size_t> dim = std::nullopt) {
  if (!dim) {
    return t.f().all_reduce(t.contiguous().view({t.size()}), 0);
  } else {
    return t.f().all_reduce(t, *dim);
  }
}

// Returns the smallest value in the input (optionally only along a given dimension).
template <typename T>
Tensor<T> min(const Tensor<T>& t, std::optional<std::size_t> dim = std::nullopt) {
  if (!dim) {
    return t.f().min_reduce(t.contiguous().view({t.size()}), 0);
  } else {
    return t.f().min_reduce(t, *dim);
  }
}

// Returns the largest value in the input (optionally only along a given dimension).
template <typename T>
Tensor<T> max(const Tensor<T>& t, std::optional<std::size_t> dim = std::nullopt) {
  if (!dim) {
    return t.f().max_reduce(t.contiguous().view({t.size()}), 0);
  } else {
    return t.f().max_reduce(t, *dim);
  }
}

// Returns the index of the largest value in the input along a given dimension. If no dimension is
// supplied the index into the flattened input is returned.
template <typename T>
[[nodiscard]] Tensor<T> argmax(const Tensor<T>& input,
                               std::optional<std::size_t> dim = std::nullopt) {
  if (!dim) {
    return input.f().argmax_reduce(input.contiguous().view({input.size()}), 0);
  } else {
    return input.f().argmax_reduce(input, *dim);
  }
}

}  // namespace tensor
