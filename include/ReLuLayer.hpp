#ifndef RELU_LAYER_HPP
#define RELU_LAYER_HPP

// #include "Tensor.hpp"
#include "ActivationLayer.hpp"

namespace myNN
{
  class ReLuLayer : public ActivationLayer
  {
  public:
    Tensor forward(const Tensor &x) override;
    Tensor backward(const Tensor &dOut) override;
  };

}

#endif