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
        Tensor dW_;
        Tensor dB_;

    public:
        // constructor
        DenseLayer(int nInputs, int nOutputs, bool initialiseGrads = false);

        // forward feed
        Tensor forward(const Tensor &input) const;

        // get weights
        Tensor &getWeights() { return w_; }

        // get biases
        Tensor &getBias() { return b_; }

        // get dW_
        Tensor &getdW_() { return dW_; }

        // get dB_
        Tensor &getdB_() { return dB_; }

        // RMSE easiest cost function
        float rmse(const Tensor &pred, const Tensor &target) const;

        // derivative of RMSE (weird naming)
        Tensor dL_dY(const Tensor &pred, const Tensor &target) const;

        // gradient of weights
        void dW(const Tensor &dL_dY, const Tensor &input);

        // gradient of bias
        void dB(const Tensor &dL_dY);

        // gradient of X
        Tensor dX(const Tensor &dL_dY);

        // backward prop
        Tensor backward(const Tensor &dL_dY);

        // update parameters
        void updateParameters(float lr);
    };

} // myNN

#endif