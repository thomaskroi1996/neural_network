#include "DenseLayer.hpp"
#include "Tensor.hpp"

#include <cmath>

using namespace myNN;

DenseLayer::DenseLayer(int nInputs, int nOutputs, bool initialiseGrads) : w_(Tensor({nInputs, nOutputs})),
                                                                          b_(Tensor({1, nOutputs})),
                                                                          dW_(Tensor({nInputs, nOutputs})),
                                                                          dB_(Tensor({1, nOutputs}))
{
    w_.apply([](float x)
             { return ((float)rand() / RAND_MAX - 0.5f); });

    if (initialiseGrads)
    {
        dW_.apply([](float x)
                  { return ((float)rand() / RAND_MAX - 0.5f); });
        dB_.apply([](float x)
                  { return ((float)rand() / RAND_MAX - 0.5f); });
    }
}

Tensor DenseLayer::forward(const Tensor &input) const
{
    return input.matMul(w_).addBroadcast(b_);
}

float DenseLayer::rmse(const Tensor &pred, const Tensor &target) const
{
    Tensor diff = pred - target;
    diff.apply([](float x)
               { return x * x; });
    float mse = diff.mean();
    return std::sqrt(mse);
}

Tensor DenseLayer::dL_dY(const Tensor &pred, const Tensor &target) const
{
    Tensor diff = pred - target;
    return diff.mul(2.0 / pred.size());
}

void DenseLayer::dW(const Tensor &dL_dY, const Tensor &input)
{
    dW_ = input.transpose().matMul(dL_dY);
}

void DenseLayer::dB(const Tensor &dL_dY)
{
    dB_ = dL_dY.sumRows();
}

Tensor DenseLayer::dX(const Tensor &dL_dY)
{
    return dL_dY.matMul(w_.transpose());
}

Tensor DenseLayer::backward(const Tensor &dL_dY)
{

    Tensor dX = DenseLayer::dX(dL_dY);

    return dX;
}

void DenseLayer::updateParameters(float lr)
{
    Tensor scaled_dW = dW_.mul(lr);
    w_.sub(scaled_dW);

    Tensor scaled_dB = dB_.mul(lr);
    b_.sub(scaled_dB);
}