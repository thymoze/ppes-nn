#pragma once

#include <iterator>
#include <sstream>
#include <tensor/tensor_data.hpp>
#include <type_traits>
#include <vector>

#ifndef DEFAULT_TENSOR_BACKEND
#define DEFAULT_TENSOR_BACKEND TensorBackend<T>(SimpleOps<T>())
#endif

namespace tensor {

template <typename T>
class Tensor;

template <typename T>
class TensorData;

template <typename T>
class TensorBackend;

template <typename T>
class SimpleOps;

std::string to_string(std::vector<std::size_t> x) {
  std::stringstream s;
  s << "(";
  std::copy(x.begin(), x.end(), std::ostream_iterator<std::size_t>(s, ", "));
  s << ")";
  return s.str();
}

template <typename T>
Tensor<T> make(Shape shape, std::vector<T>&& buffer, TensorBackend<T> backend) {
  auto data =
      TensorStorage<T>(std::make_shared<std::vector<T>>(std::move(buffer)), std::move(shape));
  return Tensor<T>(std::move(data), std::move(backend));
}
template <typename T>
Tensor<T> make(Shape shape, std::vector<T>&& buffer) {
  auto backend = DEFAULT_TENSOR_BACKEND;
  return make<T>(std::move(shape), std::move(buffer), std::move(backend));
}
template <typename T>
Tensor<T> make(std::vector<T> data) {
  return make<T>({data.size()}, std::move(data));
}
template <typename T>
Tensor<T> make(T val) {
  return make<T>({1}, {val});
}

template <typename T>
Tensor<T> zeros(Shape shape, TensorBackend<T> backend) {
  auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies());
  auto data = TensorStorage<T>(std::make_shared<std::vector<T>>(size, 0), std::move(shape));

  return Tensor<T>(std::move(data), std::move(backend));
}
template <typename T>
Tensor<T> zeros(Shape shape) {
  auto backend = DEFAULT_TENSOR_BACKEND;
  return zeros(std::move(shape), std::move(backend));
}

template <typename T>
Tensor<T> ones(Shape shape, TensorBackend<T> backend) {
  auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies());
  auto data = TensorStorage<T>(std::make_shared<std::vector<T>>(size, 1), std::move(shape));

  return Tensor<T>(std::move(data), std::move(backend));
}
template <typename T>
Tensor<T> ones(Shape shape) {
  auto backend = DEFAULT_TENSOR_BACKEND;
  return zeros(std::move(shape), std::move(backend));
}

template <typename T>
Tensor<T> rand(Shape shape, TensorBackend<T> backend, T low = 0, T hi = 1) {
  auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies());
  std::vector<T> data(size);
  std::generate(data.begin(), data.end(), [&low, &hi] { return nn::random::rand(low, hi); });

  auto _tensor =
      TensorStorage<T>(std::make_shared<std::vector<T>>(std::move(data)), std::move(shape));
  return Tensor<T>(std::move(_tensor), std::move(backend));
}
template <typename T>
Tensor<T> rand(Shape shape, T low = 0, T hi = 1) {
  auto backend = DEFAULT_TENSOR_BACKEND;
  return rand(std::move(shape), std::move(backend), low, hi);
}

}  // namespace tensor
