// standard C++ headers
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>

// Eigen matrix algebra library
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Matrix;  // import dense, dynamically sized Matrix type from Eigen;
                 // this is a matrix with row-major storage (http://en.wikipedia.org/wiki/Row-major_order)
                 // to meet the layout of the integrals returned by the Libint integral library

class TensorRank4 {

public:
  TensorRank4(int dim0, int dim1, int dim2, int dim3) {
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    data_.resize(dims_[0] * dims_[1] * dims_[2] * dims_[3]);
  }

  double& operator ()(int i, int j, int k, int l) {
    return data_(index(i, j, k, l));
  }

  const double& operator ()(int i, int j, int k, int l) const {
    return data_(index(i, j, k, l));
  }

private:

  int index(int i, int j, int k, int l) const {
    return i * dims_[2] * dims_[1] * dims_[0] + j * dims_[1] * dims_[0]
                                                                     + k * dims_[0] + l;
  }
  size_t dims_[4];
  Eigen::VectorXd data_;
};

class TensorRank6 {

public:
  TensorRank6(int dim0, int dim1, int dim2, int dim3, int dim4, int dim5) {
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    data_.resize(dims_[0] * dims_[1] * dims_[2] * dims_[3] * dims_[4] * dims_[5]);
  }

  double& operator ()(int i, int j, int k, int l, int m, int n) {
    return data_(index(i, j, k, l, m, n));
  }

  const double& operator ()(int i, int j, int k, int l, int m, int n) const {
    return data_(index(i, j, k, l, m, n));
  }

private:

  int index(int i, int j, int k, int l, int m, int n) const {
    return i * dims_[4] * dims_[3] * dims_[2] * dims_[1] * dims_[0] + j * dims_[3] * dims_[2] * dims_[1] * dims_[0]
                                                                     + k * dims_[2] * dims_[1] * dims_[0] + l * dims_[1] * dims_[0] + m * dims_[0] + n;
  }
  size_t dims_[6];
  Eigen::VectorXd data_;
};

