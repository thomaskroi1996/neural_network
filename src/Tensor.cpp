#include "Tensor.hpp"
#include <cassert>
#include <iostream>

using namespace myNN;

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

void Tensor::add(const Tensor &other)
{
  for (int i = 0; i < size(); i++)
  {
    data_[i] += other.data_[i];
  }
}

void Tensor::add(float a)
{
  for (int i = 0; i < size(); i++)
  {
    data_[i] += a;
  }
}

float Tensor::sum() const
{
  float s = 0;
  for (int i = 0; i < size(); i++)
  {
    s += data_[i];
  }
  return s;
}

float Tensor::mean() const
{
  return sum() / size();
}

Tensor Tensor::matMul(const Tensor &other) const
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

Tensor Tensor::transpose() const
{
  Tensor transposed({shape_[1], shape_[0]}, 0);
  for (int i = 0; i < shape_[0]; i++)
  {
    for (int j = 0; j < shape_[1]; j++)
    {
      transposed.data_[j * shape_[0] + i] = data_[i * shape_[1] + j];
    }
  }

  return transposed;
}

void Tensor::sub(const Tensor &other)
{
  for (int i = 0; i < size(); i++)
  {
    data_[i] -= other.data_[i];
  }
}

void Tensor::sub(float a)
{
  for (int i = 0; i < size(); i++)
  {
    data_[i] -= a;
  }
}

void Tensor::mul(const Tensor &other)
{
  for (int i = 0; i < size(); i++)
  {
    data_[i] *= other.data_[i];
  }
}

void Tensor::mul(float a)
{
  for (int i = 0; i < size(); i++)
  {
    data_[i] *= a;
  }
}

Tensor Tensor::mul(const Tensor &t, float a)
{
  Tensor result({t.getShape()});
  for (int i = 0; i < t.size(); i++)
  {
    result[i] = t[i] * a;
  }
}

Tensor Tensor::addBroadcast(const Tensor &other) const
{
  int M = shape_[0];
  int N = shape_[1];

  int m = other.shape_[0];
  int n = other.shape_[1];

  enum class Mode
  {
    FULL,   // shapes match  (M,N)+(M,N)
    SCALAR, // (1,1)
    ROW,    // (1,N)
    COL     // (M,1)
  } mode;

  if (m == 1 && n == 1)
  {
    mode = Mode::SCALAR;
  }
  else if (m == 1 && n == N)
  {
    mode = Mode::ROW;
  }
  else if (m == M && n == 1)
  {
    mode = Mode::COL;
  }
  else if (m == M && n == N)
  {
    mode = Mode::FULL;
  }
  else
  {
    throw std::runtime_error("Broadcast shapes not compatible");
  }

  Tensor out({M, N});

  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {

      float a = data_[i * N + j];
      float b;

      switch (mode)
      {
      case Mode::SCALAR:
        b = other.data_[0];
        break;

      case Mode::ROW:
        b = other.data_[j];
        break;

      case Mode::COL:
        b = other.data_[i];
        break;

      case Mode::FULL:
        b = other.data_[i * N + j];
        break;
      }

      out.data_[i * N + j] = a + b;
    }
  }

  return out;
}

Tensor Tensor::operator-(const Tensor &a) const
{
  Tensor result({a.getShape()});

  for (int i = 0; i < size(); i++)
  {
    result[i] = data_[i] - a[i];
  }
}

Tensor Tensor::sumRows() const
{
  Tensor result({1, shape_[1]}); // 1xN output

  for (int j = 0; j < shape_[1]; j++)
  {
    float s = 0.0f;
    for (int i = 0; i < shape_[0]; i++)
    {
      s += (*this)(i, j);
    }
    result(0, j) = s;
  }
  return result;
}