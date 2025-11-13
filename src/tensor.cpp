#include "tensor.hpp"
#include <cassert>
#include <iostream>

using namespace NeuralNetwork;

Tensor::Tensor(const std::vector<float> &data, const std::vector<int> &shape) : data_(data), shape_(shape) {};

Tensor::Tensor(const std::vector<int> &shape, float value) : shape_(shape)
{
  int tensor_size = 1;
  for (int s : shape)
    tensor_size *= s;
  data_.resize(tensor_size, value);
};

int Tensor::size() const
{
  return data_.size();
};

float Tensor::operator[](int i) const
{
  return data_[i];
}

float &Tensor::operator[](int i)
{
  return data_[i];
}

void Tensor::reshape(const std::vector<int> &new_shape)
{
  int old_size = 1;
  for (int s : shape_)
    old_size *= s;
  int new_size = 1;
  for (int s : new_shape)
    new_size *= s;
  shape_ = new_shape;
}

void Tensor::print() const
{
  std::cout << "Shape: " << shape_[0] << "," << shape_[1] << std::endl;

  for (int m = 0; m < shape_[0]; m++)
  {
    for (int n = 0; n < shape_[1]; n++)
    {
      std::cout << data_[m * shape_[1] + n] << " ";
    }
    std::cout << std::endl;
  }
}

Tensor Tensor::add(const Tensor &other)
{
  for (int i = 0; i < size(); i++)
  {
    data_[i] += other[i];
  }
}

Tensor Tensor::add(float a)
{
  for (int i = 0; i < size(); i++)
  {
    data_[i] += a;
  }
}

float Tensor::sum()
{
  float s = 0;
  for (int i = 0; i < size(); i++)
  {
    s += data_[i];
  }
  return s;
}

float Tensor::mean()
{
  return sum() / size();
}

Tensor Tensor::matmul(const Tensor &other)
{
  std::vector<int> out_shape = {shape_[0], other.shape_[1]};
  Tensor result(out_shape);

  int m = shape_[0];
  int n = other.shape_[1];
  int k = shape_[1];

  for (int i = 0; i < m; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      float sum = 0.0f;
      for (int p = 0; p < k; ++p)
      {
        sum += data_[i * k + p] * other.data_[p * n + j];
      }
      result.data_[i * n + j] = sum;
    }
  }

  return result;
}

// Tensor matmul(const Tensor &A, const Tensor &B)
// {
//   std::vector<int> out_shape = {A.shape_[0], B.shape_[1]};
//   Tensor result(out_shape);

//   int m = A.shape_[0];
//   int n = B.shape_[1];
//   int k = A.shape_[1];

//   for (int i = 0; i < m; ++i)
//   {
//     for (int j = 0; j < n; ++j)
//     {
//       float sum = 0.0f;
//       for (int p = 0; p < k; ++p)
//       {
//         sum += A.data_[i * k + p] * B.data_[p * n + j];
//       }
//       result.data_[i * n + j] = sum;
//     }
//   }

//   return result;
// }