#include "Network.hpp"

using namespace myNN;

Tensor Network::forwardPass(const Tensor &input)
{
    Tensor x = input;
    for (auto &layer : layers_)
    {
        x = layer.forward(input);
    }
    return x;
}

Tensor Network::backProp(const Tensor &dL_dY, const Tensor &lastInput)
{
    Tensor dX = dL_dY;
    for (auto layer = layers_.rbegin(); layer != layers_.rend(); ++layer)
    {
        layer->dB(dL_dY);
        layer->dW(dL_dY, lastInput);
        dX = layer->dX(dL_dY);
    }
    return dX;
}

void Network::addLayer(const DenseLayer &layer)
{
    layers_.push_back(layer);
}

void Network::updateParameters(float lr)
{
    for (auto &layer : layers_)
    {
        layer.updateParameters(lr); // issue
    }
}