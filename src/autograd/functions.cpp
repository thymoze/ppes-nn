#include "functions.hpp"

Variable operator+(const Variable& lhs, const Variable& rhs) {
    auto result = lhs.value() + rhs.value();

    auto gradFunc = [](std::vector<Variable>& inputs, double grad) {        
        inputs[0].add_grad(grad);
        inputs[1].add_grad(grad);
    };

    return Variable{
        result,
        { lhs, rhs },
        gradFunc
    };
};

Variable operator*(const Variable& lhs, const Variable& rhs) {
    auto result = lhs.value() * rhs.value();

    auto gradFunc = [](std::vector<Variable>& inputs, double grad) {        
        inputs[0].add_grad(grad * inputs[1].value());
        inputs[1].add_grad(grad * inputs[0].value());
    };

    return Variable{
        result,
        { lhs, rhs },
        gradFunc
    };
};