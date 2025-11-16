#include "Tensor.hpp"
#include "DenseLayer.hpp"
#include <iostream>
#include <vector>

using namespace myNN;

int main()
{
    DenseLayer layer(2, 3, true);
    Tensor input({1.0, 2.0, 3.0, 4.0}, {2, 2}); // 2x2 example
    Tensor output = layer.forward(input);
    output.print();

    Tensor dL_dY({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, {2, 3});
    Tensor dX = layer.backward(dL_dY);

    dX.print();

    layer.getWeights().print();
    layer.getBias().print();

    std::cout << "now updating params:" << std::endl;
    layer.updateParameters(5);

    std::cout << "updated params: " << std::endl;
    layer.getWeights().print();
    layer.getBias().print();

    return 0;
}

// TODO:
// definitely write all standalone functions too, not just inplace
// sort functions