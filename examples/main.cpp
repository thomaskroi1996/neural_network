#include "tensor.hpp"
#include <iostream>
#include <vector>

int main()
{
    NeuralNetwork::Tensor tensor({2, 3});
    NeuralNetwork::Tensor tensor1({2, 3});

    tensor.fill(2);
    tensor1.fill(3);

    tensor.print();
    tensor1.print();

    NeuralNetwork::Tensor added_tensor = tensor.add(tensor1);

    std::cout << "added tensor and tensor1 \n";
    added_tensor.print();

    std::cout << "add 5 to added_tensor\n";

    NeuralNetwork::Tensor add = added_tensor.add(5);

    float sum = add.sum();
    float mean = add.mean();

    add.print();

    std::cout << "sum: " << sum << ", mean: " << mean << "\n";
    return 0;
}