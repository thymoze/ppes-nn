#pragma once

#include <cassert>
#include <iomanip>
#include <memory>
#include <numeric>
#include <string>
#include <tensor/tensor_util.hpp>

Indices to_index(std::size_t ord, const Shape& shape) {
  Indices index;
  int remaining = ord;
  for (auto i = shape.rbegin(); i != shape.rend(); i++) {
    auto [quot, rem] = std::div(remaining, static_cast<int>(*i));
    index.push_back(rem);
    remaining = quot;
  }
  std::reverse(index.begin(), index.end());
  return index;
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

Indices broadcasted_to_index_in_shape(const Indices& index, const Shape& shape) {
  auto res = Indices(shape.size());
  std::transform(index.end() - res.size(), index.end(), shape.begin(), res.begin(),
                 [](auto ind, auto dim) { return ind < dim ? ind : 0; });
  return res;
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
        : shape_(std::move(shape)), idx_(idx), size_(size) {}

    value_type operator*() { return to_index(idx_, shape_); }

    // Prefix increment
    Iterator& operator++() {
      if (++idx_ >= size_) {
        idx_ = -1;
      }
      return *this;
    }

    // Postfix increment
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const Iterator& that) const { return this->idx_ == that.idx_; };
    bool operator!=(const Iterator& that) const { return this->idx_ != that.idx_; };

   private:
    Shape shape_;
    std::size_t idx_;
    std::size_t size_;
  };

  Iterator begin() { return Iterator(shape_, 0, size_); };
  Iterator end() { return Iterator(shape_, -1, size_); };

 private:
  Shape shape_;
  std::size_t size_;
};

template <typename T>
class TensorData {
 public:
  using DataPtr = std::shared_ptr<std::vector<T>>;

  using ref = T&;
  using const_ref = const T&;

  explicit TensorData(DataPtr data, Shape shape)
      : TensorData{std::move(data), shape_to_strides(shape), std::move(shape)} {}
  explicit TensorData(DataPtr data, Strides strides, Shape shape);

  [[nodiscard]] TensorData<T> permute(const Shape& order) const;

  ref get(const Indices& indices);
  const_ref get(const Indices& indices) const;

  [[nodiscard]] bool is_contiguous() const { return std::is_sorted(strides_.rbegin(), strides_.rend()); }
  [[nodiscard]] const DataPtr& data() const { return data_; }
  [[nodiscard]] IndicesIterator indices() const { return IndicesIterator(shape_); }
  [[nodiscard]] const Strides& strides() const { return strides_; }
  [[nodiscard]] const Shape& shape() const { return shape_; }
  [[nodiscard]] std::size_t size() const { return data_->size(); }
  [[nodiscard]] std::size_t ndims() const { return ndims_; }

  [[nodiscard]] std::string to_string() const;

 private:
  DataPtr data_;
  Strides strides_;
  Shape shape_;
  std::size_t ndims_;

  [[nodiscard]] std::size_t indices_to_position(const Indices& indices) const;
};

template <typename T>
TensorData<T>::TensorData(DataPtr data, Strides strides, Shape shape)
    : data_(std::move(data)),
      strides_(std::move(strides)),
      shape_(std::move(shape)),
      ndims_(strides_.size()) {
  assert(strides_.size() == shape_.size() && "Strides and shape must be same length.");
  assert(data_->size() ==
             std::accumulate(shape_.begin(), shape_.end(), 1U, std::multiplies<std::size_t>()) &&
         "Size of data must match shape.");
}

template <typename T>
TensorData<T> TensorData<T>::permute(const Shape& order) const {
  Shape shape;
  Strides strides;
  for (int i = 0; i < shape_.size(); i++) {
    assert(std::find(order.begin(), order.end(), i) != order.end() &&
           "Position required for each dimension.");

    auto pos = order[i];
    shape.push_back(shape_[pos]);
    strides.push_back(strides_[pos]);
  }

  return TensorData<T>{data_, strides, shape};
}

template <typename T>
typename TensorData<T>::ref TensorData<T>::get(const Indices& indices) {
  return const_cast<ref>(std::as_const(*this).get(indices));
}

template <typename T>
typename TensorData<T>::const_ref TensorData<T>::get(const Indices& indices) const {
  return data_->operator[](indices_to_position(indices));
}

template <typename T>
std::size_t TensorData<T>::indices_to_position(const Indices& indices) const {
  assert(indices.size() == ndims_ && "Requires exactly ndims indices.");

  std::size_t position = 0;
  for (std::size_t i = 0; i < ndims_; i++) {
    position += indices[i] * strides_[i];
  }
  return position;
}

template <typename T>
std::string TensorData<T>::to_string() const {
  std::stringstream res;
  for (auto&& index : indices()) {
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
      if (index[i] == shape_[i] - 1) {
        l += "]";
      } else {
        break;
      }
    }
    res << (l.empty() ? " " : l);
  }
  return res.str();
}
