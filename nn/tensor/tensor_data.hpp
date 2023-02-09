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

template <typename T, typename S = std::shared_ptr<std::vector<T>>>
class TensorStorage : public TensorData<T> {
 public:
  using DataPtr = S;

  explicit TensorStorage(DataPtr data, Shape shape)
      : TensorStorage{std::move(data), shape_to_strides(shape), std::move(shape)} {}
  explicit TensorStorage(DataPtr data, Strides strides, Shape shape);

  [[nodiscard]] TensorData<T>::ref at(std::size_t pos) override;
  [[nodiscard]] TensorData<T>::const_ref at(std::size_t pos) const override;

  [[nodiscard]] TensorData<T>::ref get(const Indices& indices) override;
  [[nodiscard]] TensorData<T>::const_ref get(const Indices& indices) const override;

  [[nodiscard]] TensorData<T>::ptr begin() override;
  [[nodiscard]] TensorData<T>::ptr end() override;
  [[nodiscard]] TensorData<T>::const_ptr begin() const override;
  [[nodiscard]] TensorData<T>::const_ptr end() const override;

  [[nodiscard]] std::size_t size() const override { return std::size(*data_); }

  [[nodiscard]] std::unique_ptr<TensorData<T>> clone() const override;
  [[nodiscard]] std::unique_ptr<TensorData<T>> permute(const Shape& order) const override;
  [[nodiscard]] std::unique_ptr<TensorData<T>> view(const Shape& shape) const override;
  [[nodiscard]] std::unique_ptr<TensorData<T>> view(const Shape& shape,
                                                    const Strides& strides) const override;

  [[nodiscard]] std::string to_string() const override;

 private:
  DataPtr data_;
};

template <typename T, typename S>
TensorStorage<T, S>::TensorStorage(DataPtr data, Strides strides, Shape shape)
    : TensorData<T>(std::move(strides), std::move(shape)), data_(std::move(data)) {
  assert(this->strides_.size() == this->shape_.size() && "Strides and shape must be same length.");
  assert(std::size(*data_) ==
             std::accumulate(this->shape_.begin(), this->shape_.end(), 1U, std::multiplies()) &&
         "Size of data must match shape.");
}

template <typename T, typename S>
std::unique_ptr<TensorData<T>> TensorStorage<T, S>::clone() const {
  return std::make_unique<TensorStorage<T, S>>(data_, this->strides_, this->shape_);
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

  return std::make_unique<TensorStorage<T, S>>(data_, strides, shape);
}

template <typename T, typename S>
std::unique_ptr<TensorData<T>> TensorStorage<T, S>::view(const Shape& shape) const {
  return std::make_unique<TensorStorage<T, S>>(data_, shape);
}

template <typename T, typename S>
std::unique_ptr<TensorData<T>> TensorStorage<T, S>::view(const Shape& shape,
                                                         const Strides& strides) const {
  return std::make_unique<TensorStorage<T, S>>(data_, strides, shape);
}

template <typename T, typename S>
typename TensorData<T>::ref TensorStorage<T, S>::at(std::size_t pos) {
  return const_cast<TensorData<T>::ref>(std::as_const(*this).at(pos));
}

template <typename T, typename S>
typename TensorData<T>::const_ref TensorStorage<T, S>::at(std::size_t pos) const {
  return (*data_)[pos];
}

template <typename T, typename S>
typename TensorData<T>::ref TensorStorage<T, S>::get(const Indices& indices) {
  return const_cast<TensorData<T>::ref>(std::as_const(*this).get(indices));
}

template <typename T, typename S>
typename TensorData<T>::const_ref TensorStorage<T, S>::get(const Indices& indices) const {
  return this->at(this->indices_to_position(indices));
}

template <typename T, typename S>
[[nodiscard]] typename TensorData<T>::ptr TensorStorage<T, S>::begin() {
  return const_cast<TensorData<T>::ptr>(std::as_const(*this).begin());
}

template <typename T, typename S>
[[nodiscard]] typename TensorData<T>::ptr TensorStorage<T, S>::end() {
  return const_cast<TensorData<T>::ptr>(std::as_const(*this).end());
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
    res << std::setw(5) << std::setprecision(4) << v;
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

}  // namespace tensor
