#include "DenseLayer.hpp"
#include "Tensor.hpp"

#include <cmath>

using namespace myNN;

DenseLayer::DenseLayer(int nInputs, int nOutputs) : w_(Tensor({nInputs, nOutputs})),
                                                    b_(Tensor({1, nOutputs}))
{
    w_.apply([](float x)
             { return ((float)rand() / RAND_MAX - 0.5f); });
}

Tensor DenseLayer::forward(const Tensor &input) const
{
    Tensor output = input.matMul(w_).addBroadcast(b_);
    return output;
}

float DenseLayer::rmse(const Tensor &pred, const Tensor &target) const
{
    Tensor diff = pred - target;
    diff.apply([](float x)
               { return x * x; });
    float mse = diff.mean();
    return std::sqrt(mse);
}

Tensor DenseLayer::costDerivative(const Tensor &pred, const Tensor &target) const
{
    Tensor diff = pred - target;
    Tensor derivative = diff.mul(diff, 2.0 / pred.size()); // ew, redo
    return derivative;
}

Tensor DenseLayer::wGradient(const Tensor &costDerivative, const Tensor &input)
{
    return input.transpose().matMul(costDerivative);
}

Tensor DenseLayer::bGradient(const Tensor &costDerivative)
{
    return costDerivative.sumRows();
}

Tensor DenseLayer::inputGradient(const Tensor &costDerivative)
{
    return costDerivative.matMul(w_.transpose());
}