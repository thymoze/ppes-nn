#pragma once

#include <nn/modules/linear.hpp>
#include <nn/modules/relu.hpp>
#include <nn/modules/sigmoid.hpp>
#include <nn/sequential.hpp>

template <typename T>class XOR_Model {
public:
 static nn::Sequential<T> create()
 {
   auto model = nn::Sequential<T>();
   model.add(nn::Linear<T>(nn::Matrix<T>{2, 3, {-3.654480, -0.046590, -1.998234, -4.197844, -2.199390, -0.135481}}, nn::Matrix<T>{1, 3, {0.502311, 0.349054, 0.435185}}));
   model.add(nn::Sigmoid<T>());
   model.add(nn::Linear<T>(nn::Matrix<T>{3, 1, {-3.662167, 2.466649, 2.545507}}, nn::Matrix<T>{1, 1, {-0.709472}}));
   return model;
 }
};
