#ifndef DENSE_LAYER
#define DENSE_LAYER

#include "Tensor.hpp"

namespace myNN
{

    class DenseLayer
    {
    private:
        Tensor w_;
        Tensor b_;

    public:
        // constructor
        DenseLayer(int nInputs, int nOutputs);

        // forward feed
        Tensor forward(const Tensor &input) const;

        // get weights
        Tensor &getWeights() { return w_; }

        // get biases
        Tensor &getBias() { return b_; }

        float rmse(const Tensor &pred, const Tensor &target) const;

        Tensor costDerivative(const Tensor &pred, const Tensor &target) const;

        Tensor wGradient(const Tensor &costDerivative, const Tensor &input);

        Tensor bGradient(const Tensor &costDerivative);

        Tensor inputGradient(const Tensor &costDerivative);
    };

} // myNN

#endif