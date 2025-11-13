#ifndef TENSOR
#define TENSOR

#include <vector>
#include <iostream>

namespace NeuralNetwork
{

  class Tensor
  {
  private:
    std::vector<float> data_;
    std::vector<int> shape_;

  public:
    // default constructor
    Tensor() = default;

    // constructor with only shape, initialise with 0.0f or specify value
    Tensor(const std::vector<int> &shape, float value = 0.0f);

    // constructor from 1D vector and shape vector
    Tensor(const std::vector<float> &data, const std::vector<int> &shape);

    // get number of total elements
    int size() const;

    // access value at i
    float operator[](int i) const;

    // mutable access value at i
    float &operator[](int i);

    // reshape Tensor
    void reshape(const std::vector<int> &new_shape);

    // print shape and 2D structure
    void print() const;

    // fill tensor with value a
    void fill(float a) { std::fill(data_.begin(), data_.end(), a); };

    // fill tensor with zeros
    inline void zeros()
    {
      fill(0.0);
    };

    // fill tensor with ones
    inline void ones()
    {
      fill(1.0);
    };

    // add two tensors and return new one
    Tensor add(const Tensor &t);

    // add float to tensor and return new tensor
    Tensor add(float a);

    // check tensor dims are same
    inline bool checkTensorDims(const Tensor &a, const Tensor &b) { return (a.shape_[0] == b.shape_[0] && a.shape_[1] == b.shape_[1]); }

    // return sum of all elements
    float sum();

    // return mean of all elements
    float mean();

    Tensor matmul(const Tensor &other);
  };

  // Tensor matmul(const Tensor &A, const Tensor &B);

}

#endif