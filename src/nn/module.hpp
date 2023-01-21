#pragma once

#include "autograd/variable.hpp"
#include <vector>

class Module {
public:

    virtual std::vector<Variable<double>> forward(const std::vector<Variable<double>>& inputs) = 0;

    std::vector<Variable<double>>& params() {
        return _params;
    }

    std::vector<Variable<double>> operator() (const std::vector<Variable<double>>& inputs) {
        return forward(inputs);
    }

    void zero_grad() {
        for (auto& param : _params) {
            param.zero_grad();
        }
    }

protected:
    std::vector<Variable<double>> _params;

    // Module();

};
