#ifndef TENSOR
#define TENSOR

#include <vector>
#include <iostream>

namespace myNN
{

  class Tensor
  {
  private:
    std::vector<float> data_;
    std::vector<int> shape_;

  public:
    // constructor with only shape, initialise with 0.0f or specify value
    Tensor(const std::vector<int> &shape, float value = 0.0f);

    // constructor from 1D vector and shape vector
    Tensor(const std::vector<float> &data, const std::vector<int> &shape);

    // for 1D
    Tensor(const std::vector<float> &data, int dim)
        : data_(data), shape_{dim,
                              1} {}

    // get number of total elements
    int size() const;

    // indexing function
    int index(int i, int j) const { return i * shape_[1] + j; }

    // access value at i
    float operator[](int i) const;

    // mutable access value at i
    float &operator[](int i);

    // use brackets for getting value at row i, column j
    float operator()(int i, int j) const { return data_[index(i, j)]; };

    // use brackets for getting value at row i, column j, mutable
    float &operator()(int i, int j) { return data_[index(i, j)]; };

    // reshape Tensor
    void reshape(const std::vector<int> &new_shape);

    // print shape and 2D structure
    void print() const;

    // fill tensor with value a
    void fill(float a) { std::fill(data_.begin(), data_.end(), a); }

    // fill tensor with zeros
    void zeros()
    {
      fill(0.0);
    }

    // fill tensor with ones
    void ones()
    {
      fill(1.0);
    }

    // add two tensors and return new one
    void add(const Tensor &t);

    // add float to tensor and return new tensor
    void add(float a);

    // subtract tensor from this one
    void sub(const Tensor &t);

    // subtract float from tensor
    void sub(float a);

    // subtract two tensors and return new one
    Tensor operator-(const Tensor &a) const;

    // multiply tensor with this one, element wise
    void mul_inplace(const Tensor &t);

    // multiply float with tensor
    void mul_inplace(float a);

    // multiply tensor with float and return new tensor
    Tensor mul(float a);

    // check tensor dims are same
    bool checkTensorDims(const Tensor &a, const Tensor &b) const { return (a.shape_[0] == b.shape_[0] && a.shape_[1] == b.shape_[1]); }

    // return sum of all elements
    float sum() const;

    // return mean of all elements
    float mean() const;

    // return matMul of this and other
    Tensor matMul(const Tensor &other) const;

    // return transposed tensor
    Tensor transpose() const;

    // apply function to tensor
    template <typename F>
    void apply(F func);

    // add broadcast option for adding vectors to rows or columns, and scalar to everything
    Tensor addBroadcast(const Tensor &other) const;

    // probably we should return only references, right?
    //  return tensor data
    std::vector<float> &getData() { return data_; }

    // return tensor shape
    const std::vector<int> &getShape() const { return shape_; }

    // return sum over rows
    Tensor sumRows() const;
  };

  template <typename F>
  void Tensor::apply(F func)
  {
    for (float &x : data_)
    {
      x = func(x);
    }
  }

} // namespace myNN

#endif