#pragma once

#include <algorithm>
#include <nn/random.hpp>
#include <optional>
#include <tensor/autodiff.hpp>
#include <tensor/tensor_data.hpp>
#include <tensor/tensor_functions.hpp>
#include <tensor/tensor_ops.hpp>

template <typename T>
class Tensor {
 public:
  explicit Tensor() = default;

  explicit Tensor(TensorData<T> data, TensorBackend<T> backend)
      : data_(std::move(data)), backend_(std::move(backend)) {}

  explicit Tensor(Tensor<T>&& t, History<T>&& history)
      : data_(std::move(t.data_)),
        backend_(std::move(t.backend_)),
        history_(std::make_shared<History<T>>(std::move(history))) {}

  static Tensor<T> make(T val) { return Tensor<T>::make({1}, {val}); }
  static Tensor<T> make(std::vector<T> data) {
    return Tensor<T>::make({data.size()}, std::move(data));
  }
  static Tensor<T> make(Shape shape, std::vector<T>&& buffer) {
    auto backend = TensorBackend<T>(SimpleOps<T>());
    return Tensor<T>::make(std::move(shape), std::move(buffer), std::move(backend));
  }
  static Tensor<T> make(Shape shape, std::vector<T>&& buffer, TensorBackend<T> backend) {
    auto data =
        TensorData<T>(std::make_shared<std::vector<T>>(std::move(buffer)), std::move(shape));
    return Tensor<T>(std::move(data), std::move(backend));
  }
  static Tensor<T> zeros(Shape shape) {
    auto backend = TensorBackend<T>(SimpleOps<T>());
    return Tensor<T>::zeros(std::move(shape), std::move(backend));
  }
  static Tensor<T> zeros(Shape shape, TensorBackend<T> backend) {
    auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies());
    auto data = TensorData<T>(std::make_shared<std::vector<T>>(size, 0), std::move(shape));

    return Tensor<T>(std::move(data), std::move(backend));
  }
  static Tensor<T> ones(Shape shape) {
    auto backend = TensorBackend<T>(SimpleOps<T>());
    return Tensor<T>::zeros(std::move(shape), std::move(backend));
  }
  static Tensor<T> ones(Shape shape, TensorBackend<T> backend) {
    auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies());
    auto data = TensorData<T>(std::make_shared<std::vector<T>>(size, 1), std::move(shape));

    return Tensor<T>(std::move(data), std::move(backend));
  }
  static Tensor<T> rand(Shape shape, T low, T hi) {
    auto backend = TensorBackend<T>(SimpleOps<T>());
    return Tensor<T>::rand(std::move(shape), std::move(backend), low, hi);
  }
  static Tensor<T> rand(Shape shape, TensorBackend<T> backend, T low, T hi) {
    auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies());
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&low, &hi] { return nn::random::rand(low, hi); });

    auto _tensor =
        TensorData<T>(std::make_shared<std::vector<T>>(std::move(data)), std::move(shape));
    return Tensor<T>(std::move(_tensor), std::move(backend));
  }

  [[nodiscard]] bool requires_grad() const { return static_cast<bool>(history_); }

  void requires_grad(bool grad) {
    if (grad) {
      history_ = std::make_shared<History<T>>();
    } else {
      history_.reset();
    }
  }

  [[nodiscard]] Tensor<T> detach() const { return Tensor<T>(data_, backend_); }

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
    assert(data_.data()->size() == 1 && "item() only works on single-item tensors.");
    return data_.data()->operator[](0);
  }

  template <typename... Idx>
  typename TensorData<T>::ref operator()(Idx... idx) {
    return data_.get({static_cast<std::size_t>(idx)...});
  }

  template <typename... Idx>
  typename TensorData<T>::const_ref operator()(Idx... idx) const {
    return data_.get({static_cast<std::size_t>(idx)...});
  }

  typename TensorData<T>::ref operator[](const Indices& idx) { return data_.get(idx); }
  typename TensorData<T>::const_ref operator[](const Indices& idx) const { return data_.get(idx); }

  [[nodiscard]] Tensor<T> contiguous() const { return Copy<T>()(*this); }
  [[nodiscard]] Tensor<T> view(const Shape& shape) const {
    std::vector<T> data(shape.size());
    std::transform(shape.begin(), shape.end(), data.begin(),
                   [](auto v) { return static_cast<T>(v); });
    return View<T>()(*this, Tensor<T>::make(std::move(data)));
  }
  [[nodiscard]] Tensor<T> reshape(const Shape& shape) const {
    if (data_.is_contiguous()) {
      return view(shape);
    } else {
      return contiguous().view(shape);
    }
  }

  [[nodiscard]] const TensorData<T>& data() const { return data_; }

  [[nodiscard]] IndicesIterator indices() const { return data_.indices(); }

  [[nodiscard]] const Shape& shape() const { return data_.shape(); }
  [[nodiscard]] std::size_t size() const { return data_.size(); }
  [[nodiscard]] std::size_t ndims() const { return data_.ndims(); }
  [[nodiscard]] std::string to_string() const { return data_.to_string(); }

 private:
  TensorData<T> data_;
  TensorBackend<T> backend_;
  std::shared_ptr<History<T>> history_ = std::shared_ptr<History<T>>();
  std::shared_ptr<std::optional<Tensor<T>>> grad_ =
      std::make_shared<std::optional<Tensor<T>>>(std::nullopt);
};

template <typename T>
void Tensor<T>::backward() {
  auto grad = Tensor<T>::make({1}, {1}, backend_);
  backpropagate(*this, grad);
}

template <typename T>
void Tensor<T>::add_grad(const Tensor<T>& x) {
  assert(is_leaf() && "Only leaves should have derivatives.");
  if (!grad_->has_value()) {
    *grad_ = Tensor<T>::zeros(data_.shape(), backend_);
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
  auto buf = Tensor<T>::zeros(broadcast_shape, backend_);
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

  assert(data_.data()->size() == buf.data_.data()->size() && "Expanding shapes failed");
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

template <typename T>
Tensor<T> sum(const Tensor<T>& t, std::optional<std::size_t> dim = std::nullopt) {
  if (!dim) {
    return Sum<T>()(t.contiguous().view({t.size()}), Tensor<T>::make(0));
  } else {
    return Sum<T>()(t, Tensor<T>::make(*dim));
  }
}

template <typename T>
Tensor<T> mean(const Tensor<T>& t, std::optional<std::size_t> dim = std::nullopt) {
  if (!dim) {
    return sum(t, dim) / Tensor<T>::make(t.size());
  } else {
    return sum(t, dim) / Tensor<T>::make(t.shape()[*dim]);
  }
}

template <typename T>
Tensor<T> relu(const Tensor<T>& t) {
  return ReLU<T>()(t);
}

template <typename T>
Tensor<T> sigmoid(const Tensor<T>& t) {
  return Sigmoid<T>()(t);
}

template <typename T>
Tensor<T> matmul(const Tensor<T>& lhs, const Tensor<T>& rhs) {
  return MatMul<T>()(lhs, rhs);
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, const Tensor<T>& t) {
  stream << t.to_string();
  return stream;
}
