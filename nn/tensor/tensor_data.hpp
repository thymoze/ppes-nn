#pragma once

#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

namespace tensor {

using Strides = std::vector<std::size_t>;
using Shape = std::vector<std::size_t>;
using Indices = std::vector<std::size_t>;

std::string to_string(const std::vector<std::size_t>& x) {
  std::stringstream s;
  s << "(";
  std::copy(x.begin(), x.end(), std::ostream_iterator<std::size_t>(s, ", "));
  s << ")";
  return s.str();
}

void to_index(int ord, const Shape& shape, Indices& out_index) {
  out_index.clear();
  out_index.reserve(shape.size());
  int remaining = ord;
  for (auto i = shape.rbegin(); i != shape.rend(); i++) {
    auto [quot, rem] = std::div(remaining, static_cast<int>(*i));
    out_index.push_back(rem);
    remaining = quot;
  }
  std::reverse(out_index.begin(), out_index.end());
}

Strides shape_to_strides(const Shape& shape) {
  Strides strides = {1};
  std::size_t offset = 1;
  for (auto i = shape.rbegin(); i != shape.rend() - 1; i++) {
    offset *= *i;
    strides.push_back(offset);
  }
  std::reverse(strides.begin(), strides.end());
  return strides;
};

void broadcasted_to_index_in_shape(const Indices& index, const Shape& shape, Indices& out_index) {
  out_index.clear();
  out_index.reserve(shape.size());
  std::transform(index.end() - shape.size(), index.end(), shape.begin(),
                 std::back_inserter(out_index),
                 [](auto ind, auto dim) { return ind < dim ? ind : 0; });
}

class IndicesIterator {
 public:
  explicit IndicesIterator(Shape shape)
      : shape_(std::move(shape)),
        size_(std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies())) {}

  class Iterator {
   public:
    // clang-format off
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = Indices;
    using pointer           = value_type*;
    using reference         = value_type&;
    // clang-format on

    explicit Iterator(Shape shape, std::size_t idx, std::size_t size)
        : shape_(std::move(shape)), idx_(idx), size_(size), buffer_(shape_.size()) {}

    reference operator*() {
      to_index(idx_, shape_, buffer_);
      return buffer_;
    }

    // Prefix increment
    Iterator& operator++() {
      if (++idx_ >= size_) {
        idx_ = size_;
      }
      return *this;
    }

    // Postfix increment
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    Iterator operator+(int that) {
      idx_ += that;
      return *this;
    }

    bool operator==(const Iterator& that) const { return this->idx_ == that.idx_; }
    bool operator!=(const Iterator& that) const { return this->idx_ != that.idx_; }
    bool operator<(const Iterator& that) const { return this->idx_ < that.idx_; }

   private:
    Shape shape_;
    std::size_t idx_;
    std::size_t size_;
    Indices buffer_;
  };

  Iterator begin() { return Iterator(shape_, 0, size_); };
  Iterator end() { return Iterator(shape_, size_, size_); };

 private:
  Shape shape_;
  std::size_t size_;
};

template <typename T>
class TensorData {
 public:
  using ptr = T*;
  using const_ptr = const T*;
  using ref = T&;
  using const_ref = const T&;

  virtual ~TensorData() = default;

  [[nodiscard]] virtual ref at(std::size_t pos) = 0;
  [[nodiscard]] virtual const_ref at(std::size_t pos) const = 0;

  [[nodiscard]] virtual ref get(const Indices& indices) = 0;
  [[nodiscard]] virtual const_ref get(const Indices& indices) const = 0;

  [[nodiscard]] virtual ptr begin() = 0;
  [[nodiscard]] virtual ptr end() = 0;
  [[nodiscard]] virtual const_ptr begin() const = 0;
  [[nodiscard]] virtual const_ptr end() const = 0;

  [[nodiscard]] virtual std::size_t size() const = 0;

  virtual void remove(std::size_t dim, std::size_t idx) = 0;

  [[nodiscard]] virtual std::unique_ptr<TensorData<T>> clone() const = 0;
  [[nodiscard]] virtual std::unique_ptr<TensorData<T>> permute(const Shape& order) const = 0;
  [[nodiscard]] virtual std::unique_ptr<TensorData<T>> view(const Shape& shape) const = 0;
  [[nodiscard]] virtual std::unique_ptr<TensorData<T>> view(const Shape& shape,
                                                            const Strides& strides) const = 0;

  [[nodiscard]] bool is_contiguous() const {
    return std::is_sorted(strides_.rbegin(), strides_.rend());
  }
  [[nodiscard]] IndicesIterator indices() const { return IndicesIterator(shape_); }
  [[nodiscard]] const Strides& strides() const { return strides_; }
  [[nodiscard]] const Shape& shape() const { return shape_; }
  [[nodiscard]] std::size_t ndims() const { return ndims_; }

  [[nodiscard]] virtual std::string to_string() const = 0;

 protected:
  Strides strides_;
  Shape shape_;
  std::size_t ndims_;

  TensorData(Strides strides, Shape shape)
      : strides_(std::move(strides)), shape_(std::move(shape)), ndims_(strides_.size()) {}

  [[nodiscard]] std::size_t indices_to_position(const Indices& indices) const;
};

template <typename T>
std::size_t TensorData<T>::indices_to_position(const Indices& indices) const {
  assert(indices.size() == ndims_ && "Requires exactly ndims indices.");

  std::size_t position = 0;
  for (std::size_t i = 0; i < ndims_; i++) {
    position += indices[i] * strides_[i];
  }
  return position;
}

// -------------------------------------

template <typename T, typename S>
class TensorStorage : public TensorData<T> {
 public:
  using DataPtr = S;

  explicit TensorStorage(DataPtr data, Shape shape)
      : TensorStorage{std::move(data), shape_to_strides(shape), std::move(shape)} {}
  explicit TensorStorage(DataPtr data, Strides strides, Shape shape);

  [[nodiscard]] typename TensorData<T>::ref at(std::size_t pos) override;
  [[nodiscard]] virtual typename TensorData<T>::const_ref at(std::size_t pos) const override = 0;

  [[nodiscard]] typename TensorData<T>::ref get(const Indices& indices) override;
  [[nodiscard]] typename TensorData<T>::const_ref get(const Indices& indices) const override;

  [[nodiscard]] typename TensorData<T>::ptr begin() override;
  [[nodiscard]] typename TensorData<T>::ptr end() override;
  [[nodiscard]] typename TensorData<T>::const_ptr begin() const override;
  [[nodiscard]] typename TensorData<T>::const_ptr end() const override;

  [[nodiscard]] virtual std::size_t size() const override = 0;

  virtual void remove(std::size_t dim, std::size_t idx) = 0;

  [[nodiscard]] virtual std::unique_ptr<TensorData<T>> view(
      const Shape& shape, const Strides& strides) const override = 0;
  [[nodiscard]] std::unique_ptr<TensorData<T>> permute(const Shape& order) const override;
  [[nodiscard]] std::unique_ptr<TensorData<T>> view(const Shape& shape) const override {
    return view(shape, shape_to_strides(shape));
  }
  [[nodiscard]] std::unique_ptr<TensorData<T>> clone() const override {
    return view(this->shape_, this->strides_);
  }

  [[nodiscard]] std::string to_string() const override;

 protected:
  DataPtr data_;
};

template <typename T, typename S>
TensorStorage<T, S>::TensorStorage(DataPtr data, Strides strides, Shape shape)
    : TensorData<T>(std::move(strides), std::move(shape)), data_(std::move(data)) {
  assert(this->strides_.size() == this->shape_.size() && "Strides and shape must be same length.");
}

template <typename T, typename S>
std::unique_ptr<TensorData<T>> TensorStorage<T, S>::permute(const Shape& order) const {
  Shape shape;
  Strides strides;
  for (std::size_t i = 0; i < this->shape_.size(); i++) {
    assert(std::find(order.begin(), order.end(), i) != order.end() &&
           "Position required for each dimension.");

    auto pos = order[i];
    shape.push_back(this->shape_[pos]);
    strides.push_back(this->strides_[pos]);
  }

  return view(shape, strides);
}

template <typename T, typename S>
typename TensorData<T>::ref TensorStorage<T, S>::at(std::size_t pos) {
  return const_cast<typename TensorData<T>::ref>(std::as_const(*this).at(pos));
}

template <typename T, typename S>
typename TensorData<T>::ref TensorStorage<T, S>::get(const Indices& indices) {
  return const_cast<typename TensorData<T>::ref>(std::as_const(*this).get(indices));
}

template <typename T, typename S>
typename TensorData<T>::const_ref TensorStorage<T, S>::get(const Indices& indices) const {
  return this->at(this->indices_to_position(indices));
}

template <typename T, typename S>
[[nodiscard]] typename TensorData<T>::ptr TensorStorage<T, S>::begin() {
  return const_cast<typename TensorData<T>::ptr>(std::as_const(*this).begin());
}

template <typename T, typename S>
[[nodiscard]] typename TensorData<T>::ptr TensorStorage<T, S>::end() {
  return const_cast<typename TensorData<T>::ptr>(std::as_const(*this).end());
}

template <typename T, typename S>
[[nodiscard]] typename TensorData<T>::const_ptr TensorStorage<T, S>::begin() const {
  return &this->at(0);
}

template <typename T, typename S>
[[nodiscard]] typename TensorData<T>::const_ptr TensorStorage<T, S>::end() const {
  return &this->at(this->size());
}

template <typename T, typename S>
std::string TensorStorage<T, S>::to_string() const {
  std::stringstream res;
  for (auto& index : this->indices()) {
    std::string l;
    if (std::all_of(index.begin(), index.end(), [](auto v) { return v == 0; })) {
      l.insert(0, index.size(), '[');
    } else {
      for (auto i = index.size() - 1; i >= 0; i--) {
        if (index[i] == 0) {
          l = "\n" + l.insert(0, i, ' ').insert(i, "[");
        } else {
          break;
        }
      }
    }
    res << l;
    auto v = get(index);
    if constexpr (std::is_same_v<T, char> || std::is_same_v<T, unsigned char>) {
      res << std::setw(5) << std::setprecision(4) << static_cast<int>(v);
    } else {
      res << std::setw(5) << std::setprecision(4) << v;
    }
    l = "";
    for (int i = index.size() - 1; i >= 0; i--) {
      if (index[i] == this->shape_[i] - 1) {
        l += "]";
      } else {
        break;
      }
    }
    res << (l.empty() ? " " : l);
  }
  return res.str();
}

template <typename T>
class VectorStorage : public TensorStorage<T, std::shared_ptr<std::vector<T>>> {
 public:
  using DataPtr = std::shared_ptr<std::vector<T>>;

  explicit VectorStorage(DataPtr data, Shape shape)
      : TensorStorage<T, DataPtr>(std::move(data), std::move(shape)) {}
  explicit VectorStorage(DataPtr data, Strides strides, Shape shape)
      : TensorStorage<T, DataPtr>(std::move(data), std::move(strides), std::move(shape)) {}

  using TensorData<T>::at;
  [[nodiscard]] typename TensorData<T>::const_ref at(std::size_t pos) const override {
    return (*this->data_)[pos];
  };
  [[nodiscard]] std::size_t size() const override { return this->data_->size(); };

  void remove(std::size_t dim, std::size_t idx) override {
    if (idx >= this->shape_[dim]) {
      throw std::logic_error("Cannot remove index " + std::to_string(idx) + " in dimension " +
                             std::to_string(dim) + " in tensor of shape " +
                             to_string(this->shape_));
    }

    auto stride = this->strides_[dim];
    auto size = this->size();
    auto step = dim == 0 ? size : this->strides_[dim - 1];
    std::size_t removed = 0;
    for (std::size_t start = stride * idx; start < size; start += step) {
      auto offset = this->data_->begin() + start - removed;
      this->data_->erase(offset, offset + stride);
      removed += stride;
    }

    this->shape_[dim] -= 1;
    this->strides_ = shape_to_strides(this->shape_);
  }

  [[nodiscard]] std::unique_ptr<TensorData<T>> clone() const override {
    return std::make_unique<VectorStorage<T>>(this->data_, this->strides_, this->shape_);
  }

  using TensorData<T>::view;
  [[nodiscard]] std::unique_ptr<TensorData<T>> view(const Shape& shape,
                                                    const Strides& strides) const override {
    return std::make_unique<VectorStorage<T>>(this->data_, strides, shape);
  }
};

template <typename T>
class PointerStorage : public TensorStorage<T, const T*> {
 public:
  explicit PointerStorage(const T* data, Shape shape)
      : TensorStorage<T, const T*>(std::move(data), std::move(shape)),
        size_(std::accumulate(this->shape_.begin(), this->shape_.end(), 1U, std::multiplies())) {}
  explicit PointerStorage(const T* data, Strides strides, Shape shape)
      : TensorStorage<T, const T*>(std::move(data), std::move(strides), std::move(shape)),
        size_(std::accumulate(this->shape_.begin(), this->shape_.end(), 1U, std::multiplies())) {}

  [[nodiscard]] typename TensorData<T>::const_ref at(std::size_t pos) const override {
    return *(this->data_ + pos);
  };
  [[nodiscard]] std::size_t size() const override { return size_; };

  using TensorData<T>::view;
  [[nodiscard]] std::unique_ptr<TensorData<T>> view(const Shape& shape,
                                                    const Strides& strides) const override {
    return std::make_unique<PointerStorage<T>>(this->data_, strides, shape);
  }

  void remove(std::size_t, std::size_t) override {
    throw std::logic_error("Cannot remove from pointer-backed tensor.");
  };

 private:
  std::size_t size_;
};

}  // namespace tensor
