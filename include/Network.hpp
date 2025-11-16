#ifndef NETWORK
#define NETWORK

#include "DenseLayer.hpp"

namespace myNN
{
  class Network
  {
  private:
    std::vector<DenseLayer> layers_;

  public:
    // default constructor
    Network() = default;

    // forwrard pass
    Tensor forwardPass(const Tensor &input);

    // backward propagation
    Tensor backProp(const Tensor &dL_dY, const Tensor &lastInput);

    // add a new Layer to network
    void addLayer(const DenseLayer &layer);

    // get network layers
    std::vector<DenseLayer> getLayers() { return layers_; }

    // update parameters for each layer
    void updateParameters(float lr);
  };

} // myNN

#endif