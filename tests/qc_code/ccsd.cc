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

//#include "tensor.h"

#include "tensor.h"
#include "ccsd.h"


typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Matrix;  // import dense, dynamically sized Matrix type from Eigen;
                 // this is a matrix with row-major storage (http://en.wikipedia.org/wiki/Row-major_order)
                 // to meet the layout of the integrals returned by the Libint integral library


double ccsd_energy(const int nbasis, const int ndocc, Eigen::VectorXd &E_orb, TensorRank4 &g) {

  const auto start_total = std::chrono::high_resolution_clock::now();

  const int nso = 2*nbasis;
  const int ndocc_so = 2*ndocc;

  Eigen::MatrixXd F(nbasis,nbasis);
  F = Eigen::MatrixXd::Zero(nbasis,nbasis);
  for (auto i = 0; i < nbasis; i++) {
    for (auto j = 0; j < nbasis; j++) {
      if (i == j) {
        F(i,j) = E_orb(i);
      }
    }
  }

  Eigen::MatrixXd Fso(nso,nso);
  Fso = Eigen::MatrixXd::Zero(nso,nso);

  for (auto i = 0; i < nso; i++) {
    for (auto j = 0; j < nso; j++) {
      Fso(i,j) = F(i/2,j/2) * (i%2 == j%2);
    }
  }

  TensorRank4 g_so(nso,nso,nso,nso);
  //transforming 2e integrals from spatial to spin-orbital basis
  //based on https://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project5
  for (auto p = 0; p < nso; p++) {
    for (auto q = 0; q < nso; q++) {
      for (auto r = 0; r < nso; r++) {
        for (auto s = 0; s < nso; s++) {
          double value1 = g(p/2, r/2, q/2, s/2) * (p % 2 == r % 2) * (q % 2 == s % 2);
          double value2 = g(p/2, s/2, q/2, r/2) * (p % 2 == s % 2) * (q % 2 == r % 2);
          g_so(p, q, r, s) = value1 - value2;
        }
      }
    }
  }

  double E_MP2_SO = 0.0;
  TensorRank4 t2(nso,nso,nso,nso);

  for (auto i = 0; i < ndocc_so; i++) {
    for (auto j = 0; j < ndocc_so; j++) {
      for (auto a = ndocc_so; a < nso; a++) {
        for (auto b = ndocc_so; b < nso; b++) {
          double dijab = Fso(i,i)+Fso(j,j)-Fso(a,a)-Fso(b,b);
          t2(i,a,j,b) = (g_so(i,a,j,b) - g_so(i,b,j,a))/dijab;
          E_MP2_SO += 0.25*(g_so(i,a,j,b) - g_so(i,b,j,a))*t2(i,a,j,b);
        }
      }
    }
  }

  printf("\n");
  printf("E_MP2_SO = %20.12f\n", E_MP2_SO);

  const auto stop_total = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_elapsed_total = stop_total - start_total;
  printf("Total time for MP2 spin-orbital energy evaluation module: %10.5lf sec\n", time_elapsed_total.count());

  ccsd_linear_solver(ndocc_so, nso, Fso, g_so, t2);

  return E_MP2_SO;
}

void ccsd_linear_solver(const int ndocc_so, const int nso, Eigen::MatrixXd &Fso, TensorRank4 &g_so, TensorRank4 &t2) {


  double E_CCSD = 0.0;
  Eigen::MatrixXd t1(ndocc_so, nso-ndocc_so);
  t1 = Eigen::MatrixXd::Zero(ndocc_so, nso-ndocc_so);

  int count = 0;

  while (true) {

    const auto start_it = std::chrono::high_resolution_clock::now();
    count++;

    TensorRank4 sigma2 = get_sigma2(ndocc_so, nso, Fso, g_so, t1, t2);

    Eigen::MatrixXd sigma1 = get_sigma1(ndocc_so, nso, Fso, g_so, t1, t2);

    TensorRank4 dt2 = get_double_amplitude_increment(sigma2, Fso, ndocc_so, nso);

    Eigen::MatrixXd dt1 = get_single_amplitude_increment(sigma1, Fso, ndocc_so, nso);

    t2 = update_double_amplitudes(t2, dt2, ndocc_so, nso);

    t1 = update_single_amplitudes(t1, dt1, ndocc_so, nso);

    E_CCSD = get_ccsd_energy(ndocc_so, nso, Fso, t1, t2, g_so);
    const auto stop_it = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_elapsed_it = stop_it - start_it;

    if (count == 1) {
      printf(" Iter          E_CCSD     time per iteration/sec \n");}
    printf("  %02d %20.12f     %10.5lf\n", count, E_CCSD, time_elapsed_it.count());

    double max_d = max_abs_sigma2(sigma2, ndocc_so, nso);
    double max_s = max_abs_sigma1(sigma1, ndocc_so, nso);

    if (max_d < 1e-5 && max_s < 1e-5)
      break;
  }

  printf("E_CCSD = %20.12f\n", E_CCSD);

  const auto start = std::chrono::high_resolution_clock::now();

  double E_T = triples_energy(ndocc_so, nso, Fso, t1, t2, g_so);

  const auto stop = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_elapsed = stop - start;

  printf("E_(T)  = %20.12f\n", E_T);
  printf("Total time for computation of (T): %10.5lf sec\n", time_elapsed.count());

  double E_T_lt = lt_triples_energy(ndocc_so, nso, Fso, t1, t2, g_so, E_T);

}

double max_abs_sigma2(TensorRank4 &sigma2, const int ndocc, const int nso) {
  double max = 0.0;

  for (auto i = 0; i < ndocc; i++) {
    for (auto j = 0; j < ndocc; j++) {
      for (auto a = ndocc; a < nso; a++) {
        for (auto b = ndocc; b < nso; b++) {
          max = std::max(max, std::abs(sigma2(i, a, j, b)));
        }
      }
    }
  }

  return max;
}

double max_abs_sigma1(Eigen::MatrixXd &sigma1, const int ndocc, const int nso) {
  double max = 0.0;

  for (auto i = 0; i < ndocc; i++) {
    for (auto a = ndocc; a < nso; a++) {
      max = std::max(max, std::abs(sigma1(i, a-ndocc)));
    }
  }

  return max;
}

double get_ccsd_energy(const int ndocc, const int nso, Eigen::MatrixXd &f, Eigen::MatrixXd &t1, TensorRank4 &t2, TensorRank4 &g) {

  double E_CCSD = 0.0;

  for (auto i = 0; i < ndocc; i++) {
    for (auto j = 0; j < ndocc; j++) {
      for (auto a = ndocc; a < nso; a++) {
        for (auto b = ndocc; b < nso; b++) {
          E_CCSD += f(i,a)*t1(i,a-ndocc) + 0.25*(g(i,a,j,b)-g(i,b,j,a))*t2(i,a,j,b)
              + 0.5*(g(i,a,j,b)-g(i,b,j,a))*t1(i,a-ndocc)*t1(j,b-ndocc);
        }
      }
    }
  }

  return E_CCSD;
}
TensorRank4 update_double_amplitudes(TensorRank4 &t2, TensorRank4 &dt2, const int ndocc, const int nso) {

  for (auto i = 0; i < ndocc; i++) {
    for (auto j = 0; j < ndocc; j++) {
      for (auto a = ndocc; a < nso; a++) {
        for (auto b = ndocc; b < nso; b++) {
          t2(i,a,j,b) = t2(i,a,j,b) + dt2(i,a,j,b);
        }
      }
    }
  }

  return t2;
}


Eigen::MatrixXd update_single_amplitudes(Eigen::MatrixXd &t1, Eigen::MatrixXd &dt1, const int ndocc, const int nso) {

  for (auto i = 0; i < ndocc; i++) {
    for (auto a = ndocc; a < nso; a++) {
      t1(i,a-ndocc) = t1(i,a-ndocc) + dt1(i,a-ndocc);
    }
  }

  return t1;
}


TensorRank4 get_double_amplitude_increment(TensorRank4 &sigma2, Eigen::MatrixXd &F, const int ndocc, const int nso) {

  TensorRank4 dt2(nso,nso,nso,nso);

  for (auto i = 0; i < ndocc; i++) {
    for (auto j = 0; j < ndocc; j++) {
      for (auto a = ndocc; a < nso; a++) {
        for (auto b = ndocc; b < nso; b++) {
          double dijab = F(a,a)+F(b,b)-F(i,i)-F(j,j);
          dt2(i,a,j,b) = -sigma2(i,a,j,b)/dijab;
        }
      }
    }
  }

  return dt2;
}

Eigen::MatrixXd get_single_amplitude_increment(Eigen::MatrixXd &sigma1, Eigen::MatrixXd &F, const int ndocc, const int nso) {

  Eigen::MatrixXd dt1(ndocc,nso-ndocc);

  for (auto i = 0; i < ndocc; i++) {
    for (auto a = ndocc; a < nso; a++) {
      double dia = F(a,a)-F(i,i);
      dt1(i,a-ndocc) = -sigma1(i,a-ndocc)/dia;
    }
  }

  return dt1;
}

TensorRank4 get_sigma2(const int ndocc, const int nso, Eigen::MatrixXd &f, TensorRank4 &g, Eigen::MatrixXd &t1, TensorRank4 &t2) {

  TensorRank4 sigma2(nso,nso,nso,nso);

  for (auto p = 0; p < nso; p++) {
    for (auto q = 0; q < nso; q++) {
      for (auto r = 0; r < nso; r++) {
        for (auto s = 0; s < nso; s++) {
          sigma2(p,q,r,s) = 0.0;
        }
      }
    }
  }

  Eigen::MatrixXd sigma1(nso,nso);
  sigma1 = Eigen::MatrixXd::Zero(nso,nso);

  Eigen::MatrixXd Fuu(nso-ndocc,nso-ndocc);
  Fuu = Eigen::MatrixXd::Zero(nso-ndocc,nso-ndocc);

  Eigen::MatrixXd Foo(ndocc,ndocc);
  Foo = Eigen::MatrixXd::Zero(ndocc,ndocc);

  Eigen::MatrixXd Fou(ndocc,nso-ndocc);
  Fou = Eigen::MatrixXd::Zero(ndocc,nso-ndocc);

  TensorRank4 tau_t(nso,nso,nso,nso);
  for (auto i = 0; i < ndocc; i++) {
    for (auto j = 0; j < ndocc; j++) {
      for (auto a = ndocc; a < nso; a++) {
        for (auto b = ndocc; b < nso; b++) {
          tau_t(i,a,j,b) = t2(i,a,j,b) + 0.5*(t1(i,a-ndocc)*t1(j,b-ndocc) - t1(i,b-ndocc)*t1(j,a-ndocc));
        }
      }
    }
  }

  TensorRank4 tau(nso,nso,nso,nso);
  for (auto i = 0; i < ndocc; i++) {
    for (auto j = 0; j < ndocc; j++) {
      for (auto a = ndocc; a < nso; a++) {
        for (auto b = ndocc; b < nso; b++) {
          tau(i,a,j,b) = t2(i,a,j,b) + t1(i,a-ndocc)*t1(j,b-ndocc) - t1(i,b-ndocc)*t1(j,a-ndocc);
        }
      }
    }
  }

  for (auto a = ndocc; a < nso; a++) {
    for (auto e = ndocc; e < nso; e++) {

      Fuu(a-ndocc,e-ndocc) += f(a,e) - (a == e)*f(a,e);

      double sum = 0.0;
      for (auto m = 0; m < ndocc; m++) {
        sum += f(m,e)*t1(m,a-ndocc);
      }
      Fuu(a-ndocc,e-ndocc) += -0.5*sum;

      sum = 0.0;
      for (auto m = 0; m < ndocc; m++) {
        for (auto f = ndocc; f < nso; f++) {
          sum += t1(m,f-ndocc)*(g(m,f,a,e)-g(m,e,a,f));
        }
      }
      Fuu(a-ndocc,e-ndocc) += sum;

      sum = 0.0;
      for (auto m = 0; m < ndocc; m++) {
        for (auto n = 0; n < ndocc; n++) {
          for (auto f = ndocc; f < nso; f++) {
            sum += tau_t(m,a,n,f)*(g(m,e,n,f)-g(m,f,n,e));
          }
        }
      }
      Fuu(a-ndocc,e-ndocc) += -0.5*sum;
    }
  }

  for (auto m = 0; m < ndocc; m++) {
    for (auto i = 0; i < ndocc; i++) {

      Foo(m,i) += f(m,i) - (m == i)*f(m,i);

      double sum = 0.0;
      for (auto e = ndocc; e < nso; e++) {
        sum += t1(i,e-ndocc)*f(m,e);
      }
      Foo(m,i) += 0.5*sum;

      sum = 0.0;
      for (auto e = ndocc; e < nso; e++) {
        for (auto n = 0; n < ndocc; n++) {
          sum += t1(n,e-ndocc)*(g(m,i,n,e) - g(m,e,n,i));
        }
      }
      Foo(m,i) += sum;

      sum = 0.0;
      for (auto n = 0; n < ndocc; n++) {
        for (auto e = ndocc; e < nso; e++) {
          for (auto f = ndocc; f < nso; f++) {
            sum += tau_t(i,e,n,f)*(g(m,e,n,f)-g(m,f,n,e));
          }
        }
      }
      Foo(m,i) += 0.5*sum;

    }
  }

  for (auto m = 0; m < ndocc; m++) {
    for (auto e = ndocc; e < nso; e++) {

      Fou(m,e-ndocc) += f(m,e);

      double sum = 0.0;
      for (auto n = 0; n < ndocc; n++) {
        for (auto f = ndocc; f < nso; f++) {
          sum += t1(n,f-ndocc)*(g(m,e,n,f)-g(m,f,n,e));
        }
      }
      Fou(m,e-ndocc) += sum;
    }
  }

  TensorRank4 Woo(ndocc,ndocc,ndocc,ndocc);

  for (auto m = 0; m < ndocc; m++) {
    for (auto n = 0; n < ndocc; n++) {
      for (auto i = 0; i < ndocc; i++) {
        for (auto j = 0; j < ndocc; j++) {
          Woo(m,n,i,j) = 0.0;
        }
      }
    }
  }

  for (auto m = 0; m < ndocc; m++) {
    for (auto n = 0; n < ndocc; n++) {
      for (auto i = 0; i < ndocc; i++) {
        for (auto j = 0; j < ndocc; j++) {

          Woo(m,n,i,j) += g(m,i,n,j)-g(m,j,n,i);

          double sum = 0.0;
          for (auto e = ndocc; e < nso; e++) {
            sum += t1(j,e-ndocc)*(g(m,i,n,e)-g(m,e,n,i));
          }
          Woo(m,n,i,j) += sum;

          sum = 0.0;
          for (auto e = ndocc; e < nso; e++) {
            sum += t1(i,e-ndocc)*(g(m,j,n,e)-g(m,e,n,j));
          }
          Woo(m,n,i,j) += -sum;

          sum = 0.0;
          for (auto e = ndocc; e < nso; e++) {
            for (auto f = ndocc; f < nso; f++) {
              sum += tau(i,e,j,f)*(g(m,e,n,f)-g(m,f,n,e));
            }
          }
          Woo(m,n,i,j) += 0.25*sum;

        }
      }
    }
  }

  TensorRank4 Wuu(nso-ndocc,nso-ndocc,nso-ndocc,nso-ndocc);
  for (auto a = ndocc; a < nso; a++) {
    for (auto b = ndocc; b < nso; b++) {
      for (auto e = ndocc; e < nso; e++) {
        for (auto f = ndocc; f < nso; f++) {
          Wuu(a-ndocc,b-ndocc,e-ndocc,f-ndocc) = 0.0;
        }
      }
    }
  }

  for (auto a = ndocc; a < nso; a++) {
    for (auto b = ndocc; b < nso; b++) {
      for (auto e = ndocc; e < nso; e++) {
        for (auto f = ndocc; f < nso; f++) {

          Wuu(a-ndocc,b-ndocc,e-ndocc,f-ndocc) += g(a,e,b,f) - g(a,f,b,e);

          double sum = 0.0;
          for (auto m = 0; m < ndocc; m++) {
            sum += t1(m,b-ndocc)*(g(a,e,m,f)-g(a,f,m,e));
          }
          Wuu(a-ndocc,b-ndocc,e-ndocc,f-ndocc) += -sum;

          sum = 0.0;
          for (auto m = 0; m < ndocc; m++) {
            sum += t1(m,a-ndocc)*(g(b,e,m,f)-g(b,f,m,e));
          }
          Wuu(a-ndocc,b-ndocc,e-ndocc,f-ndocc) += sum;

          sum = 0.0;
          for (auto m = 0; m < ndocc; m++) {
            for (auto n = 0; n < ndocc; n++) {
              sum += tau(m,a,n,b)*(g(m,e,n,f)-g(m,f,n,e));
            }
          }
          Wuu(a-ndocc,b-ndocc,e-ndocc,f-ndocc) += 0.25*sum;

        }
      }
    }
  }

  TensorRank4 Wou(ndocc,nso-ndocc,nso-ndocc,ndocc);
  for (auto m = 0; m < ndocc; m++) {
    for (auto b = ndocc; b < nso; b++) {
      for (auto e = ndocc; e < nso; e++) {
        for (auto j = 0; j < ndocc; j++) {
          Wou(m,b-ndocc,e-ndocc,j) = 0.0;
        }
      }
    }
  }


  for (auto m = 0; m < ndocc; m++) {
    for (auto b = ndocc; b < nso; b++) {
      for (auto e = ndocc; e < nso; e++) {
        for (auto j = 0; j < ndocc; j++) {

          Wou(m,b-ndocc,e-ndocc,j) += g(m,e,b,j)-g(m,j,b,e);

          double sum = 0.0;
          for (auto f = ndocc; f < nso; f++) {
            sum += t1(j,f-ndocc)*(g(m,e,b,f)-g(m,f,b,e));
          }
          Wou(m,b-ndocc,e-ndocc,j) += sum;

          sum = 0.0;
          for (auto n = 0; n < ndocc; n++) {
            sum += t1(n,b-ndocc)*(g(m,e,n,j)-g(m,j,n,e));
          }
          Wou(m,b-ndocc,e-ndocc,j) += -sum;

          sum = 0.0;
          for (auto n = 0; n < ndocc; n++) {
            for (auto f = ndocc; f < nso; f++) {
              sum += (0.5*t2(j,f,n,b)+t1(j,f-ndocc)*t1(n,b-ndocc))*(g(m,e,n,f)-g(m,f,n,e));
            }
          }
          Wou(m,b-ndocc,e-ndocc,j) += -sum;

        }
      }
    }
  }


  for (auto i = 0; i < ndocc; i++) {
    for (auto j = 0; j < ndocc; j++) {
      for (auto a = ndocc; a < nso; a++) {
        for (auto b = ndocc; b < nso; b++) {

          //term1
          sigma2(i,a,j,b) += -t2(i,a,j,b)*(f(i,i)+f(j,j)-f(a,a)-f(b,b));

          //term2
          sigma2(i,a,j,b) += g(i,a,j,b)-g(i,b,j,a);

          //term3 +
          double sum = 0.0;
          for (auto e = ndocc; e < nso; e++) {

            double sum2 = 0.0;
            for (auto m = 0; m < ndocc; m++) {
              sum2 += t1(m,b-ndocc)*Fou(m,e-ndocc);
            }

            sum += t2(i,a,j,e)*(Fuu(b-ndocc,e-ndocc)-0.5*sum2);

          }
          sigma2(i,a,j,b) += sum;

          //term3 -
          sum = 0.0;
          for (auto e = ndocc; e < nso; e++) {

            double sum2 = 0.0;
            for (auto m = 0; m < ndocc; m++) {
              sum2 += t1(m,a-ndocc)*Fou(m,e-ndocc);
            }

            sum += t2(i,b,j,e)*(Fuu(a-ndocc,e-ndocc)-0.5*sum2);

          }
          sigma2(i,a,j,b) += -sum;

          //term4 +
          sum = 0.0;
          for (auto m = 0; m < ndocc; m++) {

            double sum2 = 0.0;
            for (auto e = ndocc; e < nso; e++) {
              sum2 += t1(j,e-ndocc)*Fou(m,e-ndocc);
            }
            sum += t2(i,a,m,b)*(Foo(m,j) + 0.5*sum2);
          }
          sigma2(i,a,j,b) += -sum;


          //term4 -
          sum = 0.0;
          for (auto m = 0; m < ndocc; m++) {

            double sum2 = 0.0;
            for (auto e = ndocc; e < nso; e++) {
              sum2 += t1(i,e-ndocc)*Fou(m,e-ndocc);
            }
            sum += t2(j,a,m,b)*(Foo(m,i) + 0.5*sum2);
          }
          sigma2(i,a,j,b) += sum;

          //term5
          sum = 0.0;
          for (auto m = 0; m < ndocc; m++) {
            for (auto n = 0; n < ndocc; n++) {
              sum += tau(m,a,n,b)*Woo(m,n,i,j);
            }
          }
          sigma2(i,a,j,b) += 0.5*sum;

          //term6
          sum = 0.0;
          for (auto e = ndocc; e < nso; e++) {
            for (auto f = ndocc; f < nso; f++) {
              sum += tau(i,e,j,f)*Wuu(a-ndocc,b-ndocc,e-ndocc,f-ndocc);
            }
          }
          sigma2(i,a,j,b) += 0.5*sum;

          //term7 1
          sum = 0.0;
          for (auto m = 0; m < ndocc; m++) {
            for (auto e = ndocc; e < nso; e++) {
              sum += t2(i,a,m,e)*Wou(m,b-ndocc,e-ndocc,j) -
                  t1(i,e-ndocc)*t1(m,a-ndocc)*(g(m,e,b,j)-g(m,j,b,e));
            }
          }
          sigma2(i,a,j,b) += sum;
          //term7 2
          sum = 0.0;
          for (auto m = 0; m < ndocc; m++) {
            for (auto e = ndocc; e < nso; e++) {
              sum += t2(j,a,m,e)*Wou(m,b-ndocc,e-ndocc,i) -
                  t1(j,e-ndocc)*t1(m,a-ndocc)*(g(m,e,b,i)-g(m,i,b,e));
            }
          }
          sigma2(i,a,j,b) += -sum;
          //term7 3
          sum = 0.0;
          for (auto m = 0; m < ndocc; m++) {
            for (auto e = ndocc; e < nso; e++) {
              sum += t2(i,b,m,e)*Wou(m,a-ndocc,e-ndocc,j) -
                  t1(i,e-ndocc)*t1(m,b-ndocc)*(g(m,e,a,j)-g(m,j,a,e));
            }
          }
          sigma2(i,a,j,b) += -sum;
          //term7 4
          sum = 0.0;
          for (auto m = 0; m < ndocc; m++) {
            for (auto e = ndocc; e < nso; e++) {
              sum += t2(j,b,m,e)*Wou(m,a-ndocc,e-ndocc,i) -
                  t1(j,e-ndocc)*t1(m,b-ndocc)*(g(m,e,a,i)-g(m,i,a,e));
            }
          }
          sigma2(i,a,j,b) += sum;

          //term8 +
          sum = 0.0;
          for (auto e = ndocc; e < nso; e++) {
            sum += t1(i,e-ndocc)*(g(a,e,b,j)-g(a,j,b,e));
          }
          sigma2(i,a,j,b) += sum;

          //term8 -
          sum = 0.0;
          for (auto e = ndocc; e < nso; e++) {
            sum += t1(j,e-ndocc)*(g(a,e,b,i)-g(a,i,b,e));
          }
          sigma2(i,a,j,b) += -sum;

          //term9 +
          sum = 0.0;
          for (auto m = 0; m < ndocc; m++) {
            sum += t1(m,a-ndocc)*(g(m,i,b,j)-g(m,j,b,i));
          }
          sigma2(i,a,j,b) += -sum;

          //term9 -
          sum = 0.0;
          for (auto m = 0; m < ndocc; m++) {
            sum += t1(m,b-ndocc)*(g(m,i,a,j)-g(m,j,a,i));
          }
          sigma2(i,a,j,b) += sum;

        }
      }
    }
  }

  return sigma2;
}

Eigen::MatrixXd get_sigma1(const int ndocc, const int nso, Eigen::MatrixXd &f, TensorRank4 &g, Eigen::MatrixXd &t1, TensorRank4 &t2) {

  Eigen::MatrixXd sigma1(nso,nso);
  sigma1 = Eigen::MatrixXd::Zero(nso,nso);

  Eigen::MatrixXd Fuu(nso-ndocc,nso-ndocc);
  Fuu = Eigen::MatrixXd::Zero(nso-ndocc,nso-ndocc);

  Eigen::MatrixXd Foo(ndocc,ndocc);
  Foo = Eigen::MatrixXd::Zero(ndocc,ndocc);

  Eigen::MatrixXd Fou(ndocc,nso-ndocc);
  Fou = Eigen::MatrixXd::Zero(ndocc,nso-ndocc);

  TensorRank4 tau_t(nso,nso,nso,nso);
  for (auto i = 0; i < ndocc; i++) {
    for (auto j = 0; j < ndocc; j++) {
      for (auto a = ndocc; a < nso; a++) {
        for (auto b = ndocc; b < nso; b++) {
          tau_t(i,a,j,b) = t2(i,a,j,b) + 0.5*(t1(i,a-ndocc)*t1(j,b-ndocc) - t1(i,b-ndocc)*t1(j,a-ndocc));
        }
      }
    }
  }

  TensorRank4 tau(nso,nso,nso,nso);
  for (auto i = 0; i < ndocc; i++) {
    for (auto j = 0; j < ndocc; j++) {
      for (auto a = ndocc; a < nso; a++) {
        for (auto b = ndocc; b < nso; b++) {
          tau(i,a,j,b) = t2(i,a,j,b) + t1(i,a-ndocc)*t1(j,b-ndocc) - t1(i,b-ndocc)*t1(j,a-ndocc);
        }
      }
    }
  }

  for (auto a = ndocc; a < nso; a++) {
    for (auto e = ndocc; e < nso; e++) {

      Fuu(a-ndocc,e-ndocc) += f(a,e) - (a == e)*f(a,e);

      double sum = 0.0;
      for (auto m = 0; m < ndocc; m++) {
        sum += f(m,e)*t1(m,a-ndocc);
      }
      Fuu(a-ndocc,e-ndocc) += -0.5*sum;

      sum = 0.0;
      for (auto m = 0; m < ndocc; m++) {
        for (auto f = ndocc; f < nso; f++) {
          sum += t1(m,f-ndocc)*(g(m,f,a,e)-g(m,e,a,f));
        }
      }
      Fuu(a-ndocc,e-ndocc) += sum;

      sum = 0.0;
      for (auto m = 0; m < ndocc; m++) {
        for (auto n = 0; n < ndocc; n++) {
          for (auto f = ndocc; f < nso; f++) {
            sum += tau_t(m,a,n,f)*(g(m,e,n,f)-g(m,f,n,e));
          }
        }
      }
      Fuu(a-ndocc,e-ndocc) += -0.5*sum;
    }
  }

  for (auto m = 0; m < ndocc; m++) {
    for (auto i = 0; i < ndocc; i++) {

      Foo(m,i) += f(m,i) - (m == i)*f(m,i);

      double sum = 0.0;
      for (auto e = ndocc; e < nso; e++) {
        sum += t1(i,e-ndocc)*f(m,e);
      }
      Foo(m,i) += 0.5*sum;

      sum = 0.0;
      for (auto e = ndocc; e < nso; e++) {
        for (auto n = 0; n < ndocc; n++) {
          sum += t1(n,e-ndocc)*(g(m,i,n,e) - g(m,e,n,i));
        }
      }
      Foo(m,i) += sum;

      sum = 0.0;
      for (auto n = 0; n < ndocc; n++) {
        for (auto e = ndocc; e < nso; e++) {
          for (auto f = ndocc; f < nso; f++) {
            sum += tau_t(i,e,n,f)*(g(m,e,n,f)-g(m,f,n,e));
          }
        }
      }
      Foo(m,i) += 0.5*sum;

    }
  }

  for (auto m = 0; m < ndocc; m++) {
    for (auto e = ndocc; e < nso; e++) {

      Fou(m,e-ndocc) += f(m,e);

      double sum = 0.0;
      for (auto n = 0; n < ndocc; n++) {
        for (auto f = ndocc; f < nso; f++) {
          sum += t1(n,f-ndocc)*(g(m,e,n,f)-g(m,f,n,e));
        }
      }
      Fou(m,e-ndocc) += sum;
    }
  }

  for (auto i = 0; i < ndocc; i++) {
    for (auto a = ndocc; a < nso; a++) {

      //term1
      sigma1(i,a-ndocc) += -(f(i,i)-f(a,a))*t1(i,a-ndocc);
      sigma1(i,a-ndocc) += f(i,a);

      //term2
      double sum = 0.0;
      for (auto e = ndocc; e < nso; e++) {
        sum += t1(i,e-ndocc)*Fuu(a-ndocc,e-ndocc);
      }
      sigma1(i,a-ndocc) += sum;

      //term3
      sum = 0.0;
      for (auto m = 0; m < ndocc; m++) {
        sum += t1(m,a-ndocc)*Foo(m,i);
      }
      sigma1(i,a-ndocc) += -sum;

      //term4
      sum = 0.0;
      for (auto m = 0; m < ndocc; m++) {
        for (auto e = ndocc; e < nso; e++) {
          sum += t2(i,a,m,e)*Fou(m,e-ndocc);
        }
      }
      sigma1(i,a-ndocc) += sum;

      //term5
      sum = 0.0;
      for (auto n = 0; n < ndocc; n++) {
        for (auto f = ndocc; f < nso; f++) {
          sum += t1(n,f-ndocc)*(g(n,i,a,f)-g(n,f,a,i));
        }
      }
      sigma1(i,a-ndocc) += -sum;

      //term6
      sum = 0.0;
      for (auto m = 0; m < ndocc; m++) {
        for (auto e = ndocc; e < nso; e++) {
          for (auto f = ndocc; f < nso; f++) {
            sum += t2(i,e,m,f)*(g(m,e,a,f)-g(m,f,a,e));
          }
        }
      }
      sigma1(i,a-ndocc) += -0.5*sum;

      sum = 0.0;
      for (auto m = 0; m < ndocc; m++) {
        for (auto e = ndocc; e < nso; e++) {
          for (auto n = 0; n < ndocc; n++) {
            sum += t2(m,a,n,e)*(g(n,e,m,i)-g(n,i,m,e));
          }
        }
      }
      sigma1(i,a-ndocc) += -0.5*sum;

    }
  }

  return sigma1;
}

double triples_energy(const int ndocc, const int nso, Eigen::MatrixXd &f, Eigen::MatrixXd &t1, TensorRank4 &t2, TensorRank4 &g) {

  double E_T = 0.0;

  for (auto i = 0; i < ndocc; i++) {
    for (auto j = 0; j < ndocc; j++) {
      for (auto k = 0; k < ndocc; k++) {
        for (auto a = ndocc; a < nso; a++) {
          for (auto b = ndocc; b < nso; b++) {
            for (auto c = ndocc; c < nso; c++) {
              double dijkabc = f(i,i) + f(j,j) + f(k,k) - f(a,a) - f(b,b) - f(c,c);

              double tijkabc_d = t1(i,a-ndocc)*(g(j,b,k,c)-g(j,c,k,b)) - t1(j,a-ndocc)*(g(i,b,k,c)-g(i,c,k,b))
                  - t1(k,a-ndocc)*(g(j,b,i,c)-g(j,c,i,b)) - t1(i,b-ndocc)*(g(j,a,k,c)-g(j,c,k,a))
                  + t1(j,b-ndocc)*(g(i,a,k,c)-g(i,c,k,a)) + t1(k,b-ndocc)*(g(j,a,i,c)-g(j,c,i,a))
                  - t1(i,c-ndocc)*(g(j,b,k,a)-g(j,a,k,b)) + t1(j,c-ndocc)*(g(i,b,k,a)-g(i,a,k,b))
                  + t1(k,c-ndocc)*(g(j,b,i,a)-g(j,a,i,b));

              double sum1 = 0.0;
              for (auto e = ndocc; e < nso; e++) {
                sum1 += t2(j,a,k,e)*(g(e,b,i,c)-g(e,c,i,b)) - t2(i,a,k,e)*(g(e,b,j,c)-g(e,c,j,b))
                    - t2(j,a,i,e)*(g(e,b,k,c)-g(e,c,k,b)) - t2(j,b,k,e)*(g(e,a,i,c)-g(e,c,i,a))
                    + t2(i,b,k,e)*(g(e,a,j,c)-g(e,c,j,a)) + t2(j,b,i,e)*(g(e,a,k,c)-g(e,c,k,a))
                    - t2(j,c,k,e)*(g(e,b,i,a)-g(e,a,i,b)) + t2(i,c,k,e)*(g(e,b,j,a)-g(e,a,j,b))
                    + t2(j,c,i,e)*(g(e,b,k,a)-g(e,a,k,b));
              }

              double sum2 = 0.0;
              for (auto m = 0; m < ndocc; m++) {
                sum2 += t2(i,b,m,c)*(g(m,j,a,k)-g(m,k,a,j)) - t2(j,b,m,c)*(g(m,i,a,k)-g(m,k,a,i))
                    - t2(k,b,m,c)*(g(m,j,a,i)-g(m,i,a,j)) - t2(i,a,m,c)*(g(m,j,b,k)-g(m,k,b,j))
                    + t2(j,a,m,c)*(g(m,i,b,k)-g(m,k,b,i)) + t2(k,a,m,c)*(g(m,j,b,i)-g(m,i,b,j))
                    - t2(i,b,m,a)*(g(m,j,c,k)-g(m,k,c,j)) + t2(j,b,m,a)*(g(m,i,c,k)-g(m,k,c,i))
                    + t2(k,b,m,a)*(g(m,j,c,i)-g(m,i,c,j));
              }

              double tijkabc_c = sum1 - sum2;

              E_T += 1/36.0*tijkabc_c*(tijkabc_c + tijkabc_d)/dijkabc;
            }
          }
        }
      }
    }
  }

  return E_T;

}


void gauss_quad(int N, double a, double b, Eigen::VectorXd &w, Eigen::VectorXd &x) {

  Eigen::MatrixXd J;
  J.setZero(N,N);
  for (auto i = 0; i < N; i++) {
    if (i < N-1) {
      J(i,i+1) = sqrt(1/(4-pow(i+1,-2)));
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

double lt_triples_energy(const int ndocc, const int nso, Eigen::MatrixXd &f, Eigen::MatrixXd &t1, TensorRank4 &t2, TensorRank4 &g, double E_T_can) {

  double E_T_lt = 0.0;

  double alpha = 3.0*f(ndocc,ndocc) - 3.0*f(ndocc-1,ndocc-1);
  std::cout << "alpha = " << alpha << std::endl;

  printf(" \n");
    printf(" Laplace-Transform CCSD(T): \n");
    for (auto n = 1; n <= 10; n++) {

      const auto start_n = std::chrono::high_resolution_clock::now();
      Eigen::VectorXd w(n);
      Eigen::VectorXd x(n);
      gauss_quad(n, 0, 1, w, x);


      double E_T = 0.0;
      for (auto i = 0; i < ndocc; i++) {
        for (auto j = 0; j < ndocc; j++) {
          for (auto k = 0; k < ndocc; k++) {
            for (auto a = ndocc; a < nso; a++) {
              for (auto b = ndocc; b < nso; b++) {
                for (auto c = ndocc; c < nso; c++) {
                  double dijkabc = f(i,i) + f(j,j) + f(k,k) - f(a,a) - f(b,b) - f(c,c);

                  double tijkabc_d = t1(i,a-ndocc)*(g(j,b,k,c)-g(j,c,k,b)) - t1(j,a-ndocc)*(g(i,b,k,c)-g(i,c,k,b))
                      - t1(k,a-ndocc)*(g(j,b,i,c)-g(j,c,i,b)) - t1(i,b-ndocc)*(g(j,a,k,c)-g(j,c,k,a))
                      + t1(j,b-ndocc)*(g(i,a,k,c)-g(i,c,k,a)) + t1(k,b-ndocc)*(g(j,a,i,c)-g(j,c,i,a))
                      - t1(i,c-ndocc)*(g(j,b,k,a)-g(j,a,k,b)) + t1(j,c-ndocc)*(g(i,b,k,a)-g(i,a,k,b))
                      + t1(k,c-ndocc)*(g(j,b,i,a)-g(j,a,i,b));

                  double sum1 = 0.0;
                  for (auto e = ndocc; e < nso; e++) {
                    sum1 += t2(j,a,k,e)*(g(e,b,i,c)-g(e,c,i,b)) - t2(i,a,k,e)*(g(e,b,j,c)-g(e,c,j,b))
                        - t2(j,a,i,e)*(g(e,b,k,c)-g(e,c,k,b)) - t2(j,b,k,e)*(g(e,a,i,c)-g(e,c,i,a))
                        + t2(i,b,k,e)*(g(e,a,j,c)-g(e,c,j,a)) + t2(j,b,i,e)*(g(e,a,k,c)-g(e,c,k,a))
                        - t2(j,c,k,e)*(g(e,b,i,a)-g(e,a,i,b)) + t2(i,c,k,e)*(g(e,b,j,a)-g(e,a,j,b))
                        + t2(j,c,i,e)*(g(e,b,k,a)-g(e,a,k,b));
                  }

                  double sum2 = 0.0;
                  for (auto m = 0; m < ndocc; m++) {
                    sum2 += t2(i,b,m,c)*(g(m,j,a,k)-g(m,k,a,j)) - t2(j,b,m,c)*(g(m,i,a,k)-g(m,k,a,i))
                        - t2(k,b,m,c)*(g(m,j,a,i)-g(m,i,a,j)) - t2(i,a,m,c)*(g(m,j,b,k)-g(m,k,b,j))
                        + t2(j,a,m,c)*(g(m,i,b,k)-g(m,k,b,i)) + t2(k,a,m,c)*(g(m,j,b,i)-g(m,i,b,j))
                        - t2(i,b,m,a)*(g(m,j,c,k)-g(m,k,c,j)) + t2(j,b,m,a)*(g(m,i,c,k)-g(m,k,c,i))
                        + t2(k,b,m,a)*(g(m,j,c,i)-g(m,i,c,j));
                  }

                  double tijkabc_c = sum1 - sum2;

                  double D = (f(a,a)+f(b,b)+f(c,c)-f(i,i)-f(j,j)-f(k,k));
                  double integral = 1/36.0*tijkabc_c*(tijkabc_c + tijkabc_d);
                  double exponent = D/alpha - 1.0;

                  double denominator = 0.0;
                  for (auto k = 0; k < n; k++) {
                    denominator += w(k)*pow(x(k),exponent);
                  }
                  E_T += integral*denominator;

                }
              }
            }
          }
        }
      }

      E_T = -E_T/alpha;

      const auto stop_n = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> time_elapsed_n = stop_n - start_n;

      if (n == 1) {
        printf(" Number of segments         LT_CCSD(T)     error/microHartree  time for n/sec\n");}
      printf("         %02d       %20.12f %20.12f   %10.5lf\n", n, E_T, (E_T - E_T_can)*1e6, time_elapsed_n.count());
    }

  return E_T_lt;

}



