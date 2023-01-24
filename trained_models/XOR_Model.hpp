#pragma once

#include <nn/modules/linear.hpp>
#include <nn/modules/relu.hpp>
#include <nn/modules/sigmoid.hpp>
#include <nn/sequential.hpp>

class XOR_Model {
public:
 static nn::Sequential create()
 {
   auto model = nn::Sequential();
   model.add(nn::Linear(Matrix<double>{2, 3, {0.977387, 2.804045, 2.277690, 2.972495, -3.872589, -0.594189}}, Matrix<double>{1, 3, {-0.271970, -2.090667, -0.413212}}));
   model.add(nn::Sigmoid());
   model.add(nn::Linear(Matrix<double>{3, 1, {1.993647, 2.865370, -2.309926}}, Matrix<double>{1, 1, {-0.257632}}));
   return model;
 }
};
