#include "ReLuLayer.hpp"

using namespace myNN;

Tensor ReLuLayer::forward(const Tensor &x)
{
    lastInput_ = x;
    Tensor y = x;
    y.apply([](float v)
            { return v > 0 ? v : 0.0f; });
    return y;
}

Tensor ReLuLayer::backward(const Tensor &dOut)
{
    Tensor dZ = dOut;
    for (int i = 0; i < dZ.size(); i++)
    {
        dZ[i] = (lastInput_[i] > 0) ? dOut[i] : 0.0f;
    }
    return dZ;
}