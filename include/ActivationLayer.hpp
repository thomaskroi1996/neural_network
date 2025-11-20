#ifndef ACTIVATION_LAYER
#define ACTIVATION_LAYER

#include "Tensor.hpp"

namespace myNN
{
  class ActivationLayer
  {
  protected:
    Tensor lastInput_;

  public:
    virtual ~ActivationLayer() = default;
    virtual Tensor forward(const Tensor &x) = 0;
    virtual Tensor backward(const Tensor &out) = 0;
  };

} // namespace MyNN

#endif