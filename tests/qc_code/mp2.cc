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

#include "tensor.h"

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Matrix;  // import dense, dynamically sized Matrix type from Eigen;
                 // this is a matrix with row-major storage (http://en.wikipedia.org/wiki/Row-major_order)
                 // to meet the layout of the integrals returned by the Libint integral library

TensorRank4 ao_to_mo_integral_transform(const int nbasis, const int ndocc, const Eigen::MatrixXd &C,
    const TensorRank4 &g) {

  TensorRank4 g1(nbasis,nbasis,nbasis,nbasis);
  for (auto s = ndocc; s < nbasis; s++) {
    for (auto mu = 0; mu < nbasis; mu++) {
      for (auto nu = 0; nu < nbasis; nu++) {
        for (auto rho = 0; rho < nbasis; rho++) {
          double integral = 0.0;
          for (auto sigma = 0; sigma < nbasis; sigma++) {
            integral += C(sigma,s)*g(mu,nu,rho,sigma);
          }
          g1(mu,nu,rho,s) = integral;
        }
      }
    }
  }

  TensorRank4 g2(nbasis,nbasis,nbasis,nbasis);
  for (auto s = ndocc; s < nbasis; s++) {
    for (auto r = 0; r < ndocc; r++) {
      for (auto mu = 0; mu < nbasis; mu++) {
        for (auto nu = 0; nu < nbasis; nu++) {
          double integral = 0.0;
          for (auto rho = 0; rho < nbasis; rho++) {
            integral += C(rho,r)*g1(mu,nu,rho,s);
          }
          g2(mu,nu,r,s) = integral;
        }
      }
    }
  }

  TensorRank4 g3(nbasis,nbasis,nbasis,nbasis);
  for (auto s = ndocc; s < nbasis; s++) {
    for (auto r = 0; r < ndocc; r++) {
      for (auto q = ndocc; q < nbasis; q++) {
        for (auto mu = 0; mu < nbasis; mu++) {
          double integral = 0.0;
          for (auto nu = 0; nu < nbasis; nu++) {
            integral += C(nu,q)*g2(mu,nu,r,s);
          }
          g3(mu,q,r,s) = integral;
        }
      }
    }
  }


  TensorRank4 g_mo(nbasis,nbasis,nbasis,nbasis);
  for (auto s = ndocc; s < nbasis; s++) {
    for (auto r = 0; r < ndocc; r++) {
      for (auto q = ndocc; q < nbasis; q++) {
        for (auto p = 0; p < ndocc; p++) {
          double integral = 0.0;
          for (auto mu = 0; mu < nbasis; mu++) {
            integral += C(mu,p)*g3(mu,q,r,s);
          }
          g_mo(p,q,r,s) = integral;
        }
      }
    }
  }

  return g_mo;
}


double mp2_energy(const int nbasis, const int ndocc, Eigen::VectorXd &E_orb, TensorRank4 &g) {

  double E_MP2 = 0.0;

  for (auto i = 0; i < ndocc; i++) {
    for (auto j = 0; j < ndocc; j++) {
      for (auto a = ndocc; a < nbasis; a++) {
        for (auto b = ndocc; b < nbasis; b++) {
          E_MP2 += g(i,a,j,b)*(2.0*g(i,a,j,b) - 1.0*g(i,b,j,a))/(E_orb(i)+E_orb(j)-E_orb(a)-E_orb(b));
        }
      }
    }
  }

  return E_MP2;
}


void gauss_quadrature(int N, double a, double b, Eigen::VectorXd &w, Eigen::VectorXd &x) {

  Eigen::MatrixXd J;
  J.setZero(N,N);
  for (auto i = 0; i < N; i++) {
    if (i < N-1) {
      J(i,i+1) = sqrt(1/(4-pow(i+1,-2)));
    //} else if (i > 0){
      //J(i,i-1) = sqrt(1/(4-pow(i,-2)));
    }
  }
  Eigen::MatrixXd Jfin = J+J.transpose();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Jfin);
  x = es.eigenvalues();
  Eigen::MatrixXd V = es.eigenvectors();

  for (int i = 0; i < N; i++) {
    w(i) = 0.5*2.0*(b-a)*V(0,i)*V(0,i);
    x(i) = (b-a)*0.5*x(i) + (b+a)*0.5;
  }
}

void lt_mp2_energy(const int nbasis, const int ndocc, Eigen::VectorXd E_orb, TensorRank4 &g, double E_MP2_can) {

  double E_MP2_lt = 0.0;
  int N = 4;
  //Eigen::VectorXd w(N);
  //Eigen::VectorXd x(N);

  //gauss_quadrature(N, 0, 1, w, x);

  double alpha = 2.0*E_orb(ndocc) - 2.0*E_orb(ndocc-1);
  printf(" \n");
  printf(" Laplace-Transform MP2: \n");
  for (auto n = 2; n <= 32; n+=2) {
    Eigen::VectorXd w(n);
    Eigen::VectorXd x(n);
    gauss_quadrature(n, 0, 1, w, x);

  double E_MP2 = 0.0;
  for (auto i = 0; i < ndocc; i++) {
    for (auto j = 0; j < ndocc; j++) {
      for (auto a = ndocc; a < nbasis; a++) {
        for (auto b = ndocc; b < nbasis; b++) {

          double D = (E_orb(a)+E_orb(b)-E_orb(i)-E_orb(j));
          double integral = g(i,a,j,b)*(2.0*g(i,a,j,b) - 1.0*g(i,b,j,a));
          double exponent = D/alpha - 1.0;

          double denominator = 0.0;
          for (auto k = 0; k < n; k++) {
            denominator += w(k)*pow(x(k),exponent);
          }
          E_MP2 += integral*denominator;
        }
      }
    }
  }
  E_MP2 = -E_MP2/alpha;
  if (n == 2) {
    printf(" Number of segments         LT_MP2            error(LT_MP2 - MP2_canonical)/microHatree\n");}
    printf("         %02d       %20.12f            %20.12f\n", n, E_MP2, (E_MP2 - E_MP2_can)*1e6);
  }
}

