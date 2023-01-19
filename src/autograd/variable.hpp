#pragma once

#include <memory>
#include <functional>
#include <vector>

/// A container around data to facilitate reverse-mode automatic differentiation
class Variable {
public:
    using GradFunc = std::function<void (std::vector<Variable>&, double)>;

    Variable() = default;
    Variable(double value);
    Variable(double value, std::vector<Variable> inputs, GradFunc gradFunc);

    void backward() const;

    double grad() const;

    double value() const;

    void add_grad(double grad) const;

private:
    std::vector<Variable> build() const;

    // TODO: Matrix instead of double?
    // Otherwise this shared_ptr is rather pointless
    std::shared_ptr<double> shared_value = std::make_shared<double>();

    struct SharedGrad {
        double grad = 0;
        std::vector<Variable> inputs;
        GradFunc gradFunc{nullptr};
    };

    std::shared_ptr<SharedGrad> shared_grad = std::make_shared<SharedGrad>();

};
