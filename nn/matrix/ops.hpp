#pragma once

#include <matrix/matrix.hpp>
#include <numeric>
#include <cmath>

template<typename T>
Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    assert(("Addition requires matrices to be the same size!", lhs.size() == rhs.size()));

    Matrix<T> result(lhs.rows(), lhs.cols());
    std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), result.begin(), std::plus<T>());
    return result;
}

template<typename T>
Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    assert(("Subtraction requires matrices to be the same size!", lhs.size() == rhs.size()));

    Matrix<T> result(lhs.rows(), lhs.cols());
    std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), result.begin(), std::minus<T>());
    return result;
}

template<typename T>
Matrix<T> operator*(const Matrix<T>& lhs, double scalar) {
    Matrix<T> result(lhs.rows(), lhs.cols());
    std::transform(lhs.cbegin(), lhs.cend(), result.begin(), [scalar](auto& val) { return scalar * val; });
    return result;
}

template<typename T>
Matrix<T> operator*(double scalar, const Matrix<T>& rhs) {
    return rhs * scalar;
}

template<typename T>
Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    assert(("Addition requires matrices to be the same size!", lhs.size() == rhs.size()));

    Matrix<T> result(lhs.rows(), lhs.cols());
    std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), result.begin(), std::multiplies<T>());
    return result;
}

template<typename T>
Matrix<T> operator/(const Matrix<T>& lhs, double scalar) {
    Matrix<T> result(lhs.rows(), lhs.cols());
    std::transform(lhs.cbegin(), lhs.cend(), result.begin(), [scalar](auto& val) { return val / scalar; });
    return result;
}

template<typename T>
Matrix<T> operator/(double scalar, const Matrix<T>& rhs) {
    Matrix<T> result(rhs.rows(), rhs.cols());
    std::transform(rhs.cbegin(), rhs.cend(), result.begin(), [scalar](auto& val) { return scalar / val; });
    return result;
}

template<typename T>
Matrix<T> operator/(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    assert(("Division requires matrices to be the same size!", lhs.size() == rhs.size()));

    Matrix<T> result(lhs.rows(), lhs.cols());
    std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), result.begin(), std::divides<T>());
    return result;
}

//
// ------------------------------
//

namespace m {

template<typename T>
Matrix<T> sum(const Matrix<T>& mat) {
    T val = std::reduce(mat.cbegin(), mat.cend(), 0.0);

    return Matrix<T>(1, 1, { val });
}

template<typename T>
Matrix<T> mean(const Matrix<T>& mat) {
    T count = static_cast<T>(mat.rows() * mat.cols());
    T val = std::reduce(mat.cbegin(), mat.cend(), 0.0) / count;

    return Matrix<T>(1, 1, { val });
}

template<typename T>
Matrix<T> max(const Matrix<T>& mat, T value) {
    Matrix<T> result(mat.rows(), mat.cols());
    std::transform(mat.cbegin(), mat.cend(), result.begin(), [&value](auto x) { return std::max<T>(x, value); });

    return result;
}

template<typename T>
Matrix<T> exp(const Matrix<T>& mat) {
    Matrix<T> result(mat.rows(), mat.cols());
    std::transform(mat.cbegin(), mat.cend(), result.begin(), [](auto x) { return std::exp(x); });

    return result;
}

}