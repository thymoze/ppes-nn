#pragma once

#include <nn/sequential.hpp>
#include <nn/modules/linear.hpp>
#include <nn/modules/relu.hpp>
#include <nn/modules/sigmoid.hpp>

class XOR_Model
{
public:
 static nn::Sequential create()
 {
   auto model = nn::Sequential();
   model.add(nn::Linear(Matrix<double>{2, 3, {-1.653704, -1.455190, -2.842149, 2.011447, 1.094029, 2.977342}}, Matrix<double>{1, 3, {-1.486022, -0.685267, 1.760970}}));
   model.add(nn::Sigmoid());
   model.add(nn::Linear(Matrix<double>{3, 1, {2.330987, 1.277549, -2.711515}}, Matrix<double>{1, 1, {1.455618}}));
   return model;
 }
};
