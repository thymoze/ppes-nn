#pragma once

#include <algorithm>
#include <cassert>
#include <execution>
#include <initializer_list>
#include <iostream>
#include <thread>
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

    std::vector<std::size_t> rows = std::vector<std::size_t>(result.rows());
    std::iota(rows.begin(), rows.end(), 0);

    std::for_each(std::execution::par, rows.begin(), rows.end(), [&](std::size_t i) {
      for (std::size_t j = 0; j < result.cols(); ++j) {
        T val = 0;
        for (std::size_t k = 0; k < this->cols(); ++k) {
          val += this->operator()(i, k) * rhs(k, j);
        }
        result(i, j) = val;
      }
    });
    return result;
  }

  Matrix<T> matmul_seq(const Matrix<T>& rhs) const {
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

  void delete_row(std::size_t row) {
    assert(row < rows_ && "delete_row cannot delete a row greater than the existing rows");
    rows_ -= 1;
    data_.erase(data_.begin() + (row * this->cols_), data_.begin() + ((row + 1) * this->cols_));
  }

  void delete_column(std::size_t column) {
    assert(column < cols_ &&
           "delete_column cannot delete a column greater than the existing columns");
    cols_ -= 1;
    for (std::size_t i = 0; i < rows_; ++i) {
      data_.erase(data_.begin() + (i * rows_ + column));
    }
  }

  int lowest_row_sum() {
    std::vector<int> row_sums(rows_);
    for (std::size_t row = 0; row < rows_; ++row) {
      row_sums[row] =
          std::accumulate(data_.begin() + (row * cols_), data_.begin() + ((row + 1) * cols_), 0);
    }

    auto result = std::min_element(row_sums.begin(), row_sums.end());
    return std::distance(row_sums.begin(), result);
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
