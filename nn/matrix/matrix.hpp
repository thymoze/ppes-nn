#pragma once

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>

namespace nn {

template <typename T>
class Matrix {
 public:
  using ref = T&;
  using const_ref = const T&;

  Matrix() = default;

  Matrix(std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols), data_(std::vector<T>(rows_ * cols_)) {}

  Matrix(std::size_t rows, std::size_t cols, T value)
      : rows_(rows), cols_(cols), data_(std::vector<T>(rows_ * cols_, value)) {}

  Matrix(size_t rows, size_t cols, std::initializer_list<T> init)
      : rows_(rows), cols_(cols), data_(std::vector<T>(init)) {
    assert(rows_ * cols_ == data_.size());
  }

  Matrix(size_t rows, size_t cols, std::vector<T>&& init)
      : rows_(rows), cols_(cols), data_(std::move(init)) {
    assert(rows_ * cols_ == data_.size());
  }

  Matrix(std::initializer_list<std::initializer_list<T>> init)
      : rows_(init.size()), cols_(std::empty(init) ? 0 : (*init.begin()).size()) {
    assert(std::adjacent_find(init.begin(), init.end(),
                              [](auto l, auto r) { return l.size() != r.size(); }) == init.end() &&
           "All rows must be the same length!");
    data_.reserve(rows_ * cols_);
    std::for_each(init.begin(), init.end(), [this](auto list) { data_.insert(data_.end(), list); });
  }

  template <typename It>
  Matrix(size_t rows, size_t cols, It first, It last)
      : rows_(rows), cols_(cols), data_(std::vector<T>(first, last)) {}

  std::size_t rows() const { return rows_; }

  std::size_t cols() const { return cols_; }

  std::pair<std::size_t, std::size_t> size() const { return std::make_pair(rows_, cols_); }

  ref operator()(size_t i, size_t j) {
    return const_cast<ref>(std::as_const(*this).operator()(i, j));
  }

  const_ref operator()(size_t i, size_t j) const { return data_[i * cols_ + j]; }

  Matrix<T> matmul(const Matrix<T>& rhs) const {
    assert(this->cols() == rhs.rows() &&
           "Dimension must match for matrix multiplication (MxN)*(NxK)");

    Matrix<T> result(this->rows(), rhs.cols());
    for (std::size_t i = 0; i < result.rows(); i++) {
      for (std::size_t j = 0; j < result.cols(); j++) {
        T val = 0;
        for (std::size_t k = 0; k < this->cols(); k++) {
          val = val + this->operator()(i, k) * rhs(k, j);
        }
        result(i, j) = val;
      }
    }
    return result;
  }

  Matrix<T> transpose() const {
    Matrix<T> result(cols(), rows());
    for (std::size_t i = 0; i < result.rows(); i++) {
      for (std::size_t j = 0; j < result.cols(); j++) {
        result(i, j) = this->operator()(j, i);
      }
    }
    return result;
  }

  void reshape(std::size_t rows, std::size_t cols) {
    assert(rows * cols == rows_ * cols_ && "Reshape cannot change element count");
    rows_ = rows;
    cols_ = cols;
  }

  std::vector<T>& data() { return data_; }

  auto begin() { return data_.begin(); }
  auto cbegin() const { return data_.cbegin(); }
  auto end() { return data_.end(); }
  auto cend() const { return data_.cend(); }
  auto rbegin() { return data_.rbegin(); }
  auto crbegin() const { return data_.crbegin(); }
  auto rend() { return data_.rend(); }
  auto crend() const { return data_.crend(); }

 private:
  size_t rows_, cols_;

  std::vector<T> data_;
};

}  // namespace nn

#include <matrix/ops.hpp>
