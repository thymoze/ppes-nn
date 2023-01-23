#pragma once

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>

template <typename T>
class Matrix {
 public:
  using ref = T&;
  using const_ref = const T&;

  Matrix() = default;

  Matrix(std::size_t rows, std::size_t cols)
      : _rows(rows), _cols(cols), _data(std::vector<T>(_rows * _cols)) {}

  Matrix(std::size_t rows, std::size_t cols, T value)
      : _rows(rows), _cols(cols), _data(std::vector<T>(_rows * _cols, value)) {}

  Matrix(size_t rows, size_t cols, std::initializer_list<T> init)
      : _rows(rows), _cols(cols), _data(std::vector<T>(init)) {
    assert(_rows * _cols == _data.size());
  }

  Matrix(std::initializer_list<std::initializer_list<T>> init)
      : _rows(init.size()), _cols(std::empty(init) ? 0 : (*init.begin()).size()) {
    assert(("All rows must be the same length!",
            std::adjacent_find(init.begin(), init.end(),
                               [](auto l, auto r) { return l.size() != r.size(); }) == init.end()));
    _data.reserve(_rows * _cols);
    std::for_each(init.begin(), init.end(), [this](auto list) { _data.insert(_data.end(), list); });
  }

  template <typename It>
  Matrix(size_t rows, size_t cols, It first, It last)
      : _rows(rows), _cols(cols), _data(std::vector<T>(first, last)) {}

  std::size_t rows() const { return _rows; }

  std::size_t cols() const { return _cols; }

  std::pair<std::size_t, std::size_t> size() const { return std::make_pair(_rows, _cols); }

  ref operator()(size_t i, size_t j) {
    return const_cast<ref>(std::as_const(*this).operator()(i, j));
  }

  const_ref operator()(size_t i, size_t j) const { return _data[i * _cols + j]; }

  Matrix<T> matmul(const Matrix<T>& rhs) const {
    assert(
        ("Dimension must match for matrix multiplication (MxN)*(NxK)", this->cols() == rhs.rows()));

    Matrix<T> result(this->rows(), rhs.cols());
    for (int i = 0; i < result.rows(); i++) {
      for (int j = 0; j < result.cols(); j++) {
        T val = 0;
        for (int k = 0; k < this->cols(); k++) {
          val = val + this->operator()(i, k) * rhs(k, j);
        }
        result(i, j) = val;
      }
    }
    return result;
  }

  Matrix<T> transpose() const {
    Matrix<T> result(cols(), rows());
    for (int i = 0; i < result.rows(); i++) {
      for (int j = 0; j < result.cols(); j++) {
        result(i, j) = this->operator()(j, i);
      }
    }
    return result;
  }

  std::vector<T>& data() { return _data; }

  auto begin() { return _data.begin(); }
  auto cbegin() const { return _data.cbegin(); }
  auto end() { return _data.end(); }
  auto cend() const { return _data.cend(); }
  auto rbegin() { return _data.rbegin(); }
  auto crbegin() const { return _data.crbegin(); }
  auto rend() { return _data.rend(); }
  auto crend() const { return _data.crend(); }

 private:
  size_t _rows, _cols;

  std::vector<T> _data;
};

#include <matrix/ops.hpp>
