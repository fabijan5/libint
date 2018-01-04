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

  const auto start_total = std::chrono::high_resolution_clock::now();

  TensorRank4 g1(nbasis,nbasis,nbasis,nbasis);
  for (auto s = 0; s < nbasis; s++) {
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
  for (auto s = 0; s < nbasis; s++) {
    for (auto r = 0; r < nbasis; r++) {
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
  for (auto s = 0; s < nbasis; s++) {
    for (auto r = 0; r < nbasis; r++) {
      for (auto q = 0; q < nbasis; q++) {
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
  for (auto s = 0; s < nbasis; s++) {
    for (auto r = 0; r < nbasis; r++) {
      for (auto q = 0; q < nbasis; q++) {
        for (auto p = 0; p < nbasis; p++) {
          double integral = 0.0;
          for (auto mu = 0; mu < nbasis; mu++) {
            integral += C(mu,p)*g3(mu,q,r,s);
          }
          g_mo(p,q,r,s) = integral;
        }
      }
    }
  }

  const auto stop_total = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_elapsed_total = stop_total - start_total;
  printf("Total time for 2e-integrals transformation module: %10.5lf sec\n", time_elapsed_total.count());

  return g_mo;
}


double mp2_energy(const int nbasis, const int ndocc, Eigen::VectorXd &E_orb, TensorRank4 &g) {

  const auto start_total = std::chrono::high_resolution_clock::now();

  double E_MP2 = 0.0;
  int fc = 0;
  for (auto i = fc; i < ndocc; i++) {
    for (auto j = fc; j < ndocc; j++) {
      double e_ij = 0.0;
      for (auto a = ndocc; a < nbasis; a++) {
        for (auto b = ndocc; b < nbasis; b++) {
          E_MP2 += g(i,a,j,b)*(2.0*g(i,a,j,b) - 1.0*g(i,b,j,a))/(E_orb(i)+E_orb(j)-E_orb(a)-E_orb(b));
          e_ij += g(i,a,j,b)*(2.0*g(i,a,j,b) - 1.0*g(i,b,j,a))/(E_orb(i)+E_orb(j)-E_orb(a)-E_orb(b));
        }
      }
      //std::cout << "i = " << i - fc << " j = " << j - fc << " e_ij = " << e_ij << std::endl;
    }
  }

  printf("\n");
  printf("E_MP2 = %20.12f\n", E_MP2);

  const auto stop_total = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_elapsed_total = stop_total - start_total;
  printf("Total time for MP2 energy evaluation module: %10.5lf sec\n", time_elapsed_total.count());

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


  double alpha = 2.0*E_orb(ndocc) - 2.0*E_orb(ndocc-1);
  std::cout << alpha << std::endl;
  printf(" \n");
  printf(" Laplace-Transform MP2: \n");
  for (auto n = 2; n <= 32; n+=2) {

    const auto start_n = std::chrono::high_resolution_clock::now();
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

    const auto stop_n = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_elapsed_n = stop_n - start_n;

    if (n == 2) {
      printf(" Number of segments         LT_MP2        error/microHartree  time for n/sec\n");}
    printf("         %02d       %20.12f %20.12f   %10.5lf\n", n, E_MP2, (E_MP2 - E_MP2_can)*1e6, time_elapsed_n.count());
  }
}

void lt_ao_mp2_energy(const int nbasis, const int ndocc, Eigen::VectorXd E_orb, Eigen::MatrixXd &C, TensorRank4 &g, double E_MP2_can) {


  double alpha = 2.0*E_orb(ndocc) - 2.0*E_orb(ndocc-1);
  std::cout << alpha << std::endl;
  printf(" \n");
  printf(" Laplace-Transform AO-MP2: \n");
  //for (auto n = 1; n <= 1; n+=1) {

  Eigen::MatrixXd C_loc(7,7);
  /*
  C_loc(0,0) = 0.100180; C_loc(0,1) = 1.013424; C_loc(0,2) = -0.100176; C_loc(0,3) = -0.062349; C_loc(0,4) = -0.062349; C_loc(0,5) = 0.111639; C_loc(0,6) = -0.000000;
  C_loc(1,0) = -0.667039; C_loc(1,1) = -0.067617; C_loc(1,2) = 0.667043; C_loc(1,3) = 0.245616; C_loc(1,4) = 0.245613; C_loc(1,5) = -0.669577; C_loc(1,6) = 0.000000;
  C_loc(2,0) = -0.707108; C_loc(2,1) = 0.000003; C_loc(2,2) = -0.707105; C_loc(2,3) = 0.000006; C_loc(2,4) = -0.000001; C_loc(2,5) = -0.000000; C_loc(2,6) = 0.000000;
  C_loc(3,0) = 0.000003; C_loc(3,1) = 0.000000; C_loc(3,2) = 0.000001; C_loc(3,3) = 0.429416; C_loc(3,4) = -0.429416; C_loc(3,5) = 0.000000; C_loc(3,6) = 0.919233;
  C_loc(4,0) = 0.355569; C_loc(4,1) = 0.030249; C_loc(4,2) = -0.355567; C_loc(4,3) = 0.395136; C_loc(4,4) = 0.395139; C_loc(4,5) = -0.738490; C_loc(4,6) = 0.000000;
  C_loc(5,0) = 0.107755; C_loc(5,1) = -0.001004; C_loc(5,2) = -0.107751; C_loc(5,3) = 0.550641; C_loc(5,4) = -0.089994; C_loc(5,5) = 0.709849; C_loc(5,6) = -0.732461;
  C_loc(6,0) = 0.107750; C_loc(6,1) = -0.001005; C_loc(6,2) = -0.107752; C_loc(6,3) = -0.089994; C_loc(6,4) = 0.550642; C_loc(6,5) = 0.709849; C_loc(6,6) = 0.732461;
*/

  /*
  C_loc(0,0) = 0.100180; C_loc(0,1) = 1.013424; C_loc(0,2) = -0.100176; C_loc(0,3) = -0.062349; C_loc(0,4) = -0.062349; C_loc(0,5) = 0.111639; C_loc(0,6) = -0.000000;
    C_loc(1,0) = -0.667039; C_loc(1,1) = -0.067617; C_loc(1,2) = 0.667043; C_loc(1,3) = 0.245616; C_loc(1,4) = 0.245613; C_loc(1,5) = -0.669577; C_loc(1,6) = 0.000000;
    C_loc(4,0) = -0.707108; C_loc(4,1) = 0.000003; C_loc(4,2) = -0.707105; C_loc(4,3) = 0.000006; C_loc(4,4) = -0.000001; C_loc(4,5) = -0.000000; C_loc(4,6) = 0.000000;
    C_loc(2,0) = 0.000003; C_loc(2,1) = 0.000000; C_loc(2,2) = 0.000001; C_loc(2,3) = 0.429416; C_loc(2,4) = -0.429416; C_loc(2,5) = 0.000000; C_loc(2,6) = 0.919233;
    C_loc(3,0) = 0.355569; C_loc(3,1) = 0.030249; C_loc(3,2) = -0.355567; C_loc(3,3) = 0.395136; C_loc(3,4) = 0.395139; C_loc(3,5) = -0.738490; C_loc(3,6) = 0.000000;
    C_loc(5,0) = 0.107755; C_loc(5,1) = -0.001004; C_loc(5,2) = -0.107751; C_loc(5,3) = 0.550641; C_loc(5,4) = -0.089994; C_loc(5,5) = 0.709849; C_loc(5,6) = -0.732461;
    C_loc(6,0) = 0.107750; C_loc(6,1) = -0.001005; C_loc(6,2) = -0.107752; C_loc(6,3) = -0.089994; C_loc(6,4) = 0.550642; C_loc(6,5) = 0.709849; C_loc(6,6) = 0.732461;

    Eigen::MatrixXd C_loc1(7,7);
    C_loc1(0,0) = -0.130726; C_loc1(0,1) = -0.130729; C_loc1(0,2) = -0.038894 ; C_loc1(0,3) = 0.038895; C_loc1(0,4) = 0.000138; C_loc1(0,5) = -0.111640; C_loc1(0,6) = 0.000000;
    C_loc1(1,0) = 0.551001; C_loc1(1,1) = 0.551012; C_loc1(1,2) = 0.078514; C_loc1(1,3) = -0.078515; C_loc1(1,4) = -0.000008; C_loc1(1,5) = 0.669578; C_loc1(1,6) = -0.000000;
    C_loc1(4,0) = 0.721556; C_loc1(4,1) = -0.721541; C_loc1(4,2) = -0.000002; C_loc1(4,3) = 0.000002; C_loc1(4,4) = 0.000000; C_loc1(4,5) = -0.000000; C_loc1(4,6) = 0.000000;
    C_loc1(2,0) = 0.000001; C_loc1(2,1) = 0.000001; C_loc1(2,2) = 0.407733; C_loc1(2,3) = 0.407733; C_loc1(2,4) = -0.000000; C_loc1(2,5) = 0.000000; C_loc1(2,6) = 0.919233;
    C_loc1(3,0) = -0.346760; C_loc1(3,1) = -0.346769; C_loc1(3,2) = 0.399917; C_loc1(3,3) = -0.399916; C_loc1(3,4) = 0.000004; C_loc1(3,5) = 0.738490; C_loc1(3,6) = -0.000000;
    C_loc1(5,0) = -0.115781; C_loc1(5,1) = -0.115785; C_loc1(5,2) = 0.518248; C_loc1(5,3) = 0.090040; C_loc1(5,4) = -0.000000; C_loc1(5,5) = -0.709849; C_loc1(5,6) = -0.732461;
    C_loc1(6,0) = -0.115782; C_loc1(6,1) = -0.115786; C_loc1(6,2) = -0.090040; C_loc1(6,3) = -0.518248; C_loc1(6,4) = -0.000000; C_loc1(6,5) = -0.709849; C_loc1(6,6) =  0.732461;


    Eigen::MatrixXd C_loc2(7,7);
    C_loc2(0,0) = -0.152722; C_loc2(0,1) = -0.152723; C_loc2(0,2) = -0.073618; C_loc2(0,3) = 0.073618; C_loc2(0,4) = 0.260389; C_loc2(0,5) = -0.111640; C_loc2(0,6) = 0.000000;
    C_loc2(1,0) = 0.64870; C_loc2(1,1) = 0.648708; C_loc2(1,2) = 0.215936; C_loc2(1,3) = -0.215937; C_loc2(1,4) = -0.014590; C_loc2(1,5) = 0.669578; C_loc2(1,6) = -0.000000;
    C_loc2(4,0) = 0.709295; C_loc2(4,1) = -0.709292; C_loc2(4,2) = -0.000001; C_loc2(4,3) = -0.000004; C_loc2(4,4) = 0.000000; C_loc2(4,5) = -0.000000; C_loc2(4,6) = 0.000000;
    C_loc2(2,0) = 0.000002; C_loc2(2,1) = -0.000001; C_loc2(2,2) = 0.426031; C_loc2(2,3) = 0.426031; C_loc2(2,4) = 0.000000; C_loc2(2,5) = 0.000000; C_loc2(2,6) = 0.919233;
    C_loc2(3,0) = -0.355016; C_loc2(3,1) = -0.355016; C_loc2(3,2) = 0.395523; C_loc2(3,3) = -0.395523; C_loc2(3,4) = 0.005849; C_loc2(3,5) = 0.738490; C_loc2(3,6) = -0.000000;
    C_loc2(5,0) = -0.109024; C_loc2(5,1) = -0.109026; C_loc2(5,2) = 0.545136; C_loc2(5,3) = 0.090450; C_loc2(5,4) = -0.000937; C_loc2(5,5) = -0.709849; C_loc2(5,6) = -0.732461;
    C_loc2(6,0) = -0.109026; C_loc2(6,1) = -0.109024; C_loc2(6,2) = -0.090450; C_loc2(6,3) = -0.545136; C_loc2(6,4) = -0.000937; C_loc2(6,5) = -0.709849; C_loc2(6,6) = 0.732461;

    Eigen::MatrixXd U(7,7);
    U(0,0) = -0.99585; U(0,1) = 0.06093; U(0,2) = -0.06091; U(0,3) = -0.02070; U(0,4) = 0.02070; U(0,5) = 0.00000; U(0,6) = 0.00000;
    U(1,0) = 0.07890; U(1,1) = 0.46479; U(1,2) = -0.46479; U(1,3) = -0.52996; U(1,4) = 0.52996; U(1,5) = 0.00000; U(1,6) = 0.00000;
    U(2,0) = 0.00000; U(2,1) = -0.00000; U(2,2) = -0.00000; U(2,3) = -0.70711; U(2,4) = -0.70711; U(2,5) = 0.00000; U(2,6) = 0.00000;
    U(3,0) = 0.04533; U(3,1) = 0.52939; U(3,2) = -0.52940; U(3,3) = 0.46767; U(3,4) = -0.46767; U(3,5) = 0.00000; U(3,6) = 0.00000;
    U(4,0) = -0.00001; U(4,1) = -0.70711; U(4,2) = -0.70710; U(4,3) = -0.00000; U(4,4) = 0.00000; U(4,5) = 0.00000; U(4,6) = 0.00000;
    U(5,0) = -0.00000; U(5,1) = 0.00000; U(5,2) = 0.00000; U(5,3) = 0.00000; U(5,4) = 0.00000; U(5,5) = 1.00000; U(5,6) = 0.00000;
    U(6,0) = -0.00000; U(6,1) = 0.00000; U(6,2) = 0.00000; U(6,3) = 0.00000; U(6,4) = 0.00000; U(6,5) = 0.00000; U(6,6) = 1.00000;

    std::cout << C << std::endl;*/
    int n = 16;
    const auto start_n = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd w(n);
    Eigen::VectorXd x(n);
    gauss_quadrature(n, 0, 1, w, x);

    //std::cout << x << std::endl;
    //std::cout << w << std::endl;

    double E_MP2 = 0.0;

    for (auto k = 0; k < n; k++) {
      Eigen::MatrixXd Po, Pu, T;
      Po.setZero(nbasis,nbasis);
      Pu.setZero(nbasis,nbasis);
      T.setZero(ndocc,nbasis);

      /*
      for (auto i = 0; i < ndocc; i++) {
        for (auto mu = 0; mu < nbasis; mu++) {
          double sum = 0.0;
          for (auto I = 0; I < ndocc; I++) {
            sum += U(i,I)*pow(x(k),-E_orb(I)/(2.0*alpha)-1/8.0)*C(mu,I);
          }
          T(i,mu) = sum;
        }
      }

      std::cout << T << std::endl;*/

      for (auto mu_p = 0; mu_p < nbasis; mu_p++) {
        for (auto mu = 0; mu < nbasis; mu++) {
          double sum = 0.0;
          for (auto i = 0; i < ndocc; i++) {
            sum += C(mu_p,i)*pow(x(k),-E_orb(i)/alpha - 0.25)*C(mu,i);
            //sum += C_loc1(mu_p,i)*C_loc1(mu,i);
            //sum += T(i,mu_p)*T(i,mu);
          }
          Po(mu_p,mu) = sum;
        }
      }

      for (auto mu_p = 0; mu_p < nbasis; mu_p++) {
        for (auto mu = 0; mu < nbasis; mu++) {
          double sum = 0.0;
          for (auto a = ndocc; a < nbasis; a++) {
            sum += C(mu_p,a)*pow(x(k),E_orb(a)/alpha - 0.25)*C(mu,a);
          }
          Pu(mu_p,mu) = sum;
        }
      }

      TensorRank4 g_p1(nbasis,nbasis,nbasis,nbasis);
      for (auto sigma = 0; sigma < nbasis; sigma++) {
        for (auto mu_p = 0; mu_p < nbasis; mu_p++) {
          for (auto nu_p = 0; nu_p < nbasis; nu_p++) {
            for (auto rho_p = 0; rho_p < nbasis; rho_p++) {
              double sum = 0.0;
              for (auto sigma_p = 0; sigma_p < nbasis; sigma_p++) {
                sum += g(mu_p,nu_p,rho_p,sigma_p)*Pu(sigma_p,sigma);
              }
              g_p1(mu_p,nu_p,rho_p,sigma) = sum;
            }
          }
        }
      }

      TensorRank4 g_p2(nbasis,nbasis,nbasis,nbasis);
      for (auto sigma = 0; sigma < nbasis; sigma++) {
        for (auto rho = 0; rho < nbasis; rho++) {
          for (auto mu_p = 0; mu_p < nbasis; mu_p++) {
            for (auto nu_p = 0; nu_p < nbasis; nu_p++) {
              double sum = 0.0;
              for (auto rho_p = 0; rho_p < nbasis; rho_p++) {
                sum += g_p1(mu_p,nu_p,rho_p,sigma)*Po(rho_p,rho);
              }
              g_p2(mu_p,nu_p,rho,sigma) = sum;
            }
          }
        }
      }

      TensorRank4 g_p3(nbasis,nbasis,nbasis,nbasis);
      for (auto sigma = 0; sigma < nbasis; sigma++) {
        for (auto rho = 0; rho < nbasis; rho++) {
          for (auto nu = 0; nu < nbasis; nu++) {
            for (auto mu_p = 0; mu_p < nbasis; mu_p++) {
              double sum = 0.0;
              for (auto nu_p = 0; nu_p < nbasis; nu_p++) {
                sum += g_p2(mu_p,nu_p,rho,sigma)*Pu(nu_p,nu);
              }
              g_p3(mu_p,nu,rho,sigma) = sum;
            }
          }
        }
      }

      TensorRank4 g_p(nbasis,nbasis,nbasis,nbasis);
      for (auto sigma = 0; sigma < nbasis; sigma++) {
        for (auto rho = 0; rho < nbasis; rho++) {
          for (auto nu = 0; nu < nbasis; nu++) {
            for (auto mu = 0; mu < nbasis; mu++) {
              double sum = 0.0;
              for (auto mu_p = 0; mu_p < nbasis; mu_p++) {
                sum += g_p3(mu_p,nu,rho,sigma)*Po(mu_p,mu);
              }
              g_p(mu,nu,rho,sigma) = sum;
            }
          }
        }
      }

      double E_k = 0.0;
      for (auto sigma = 0; sigma < nbasis; sigma++) {
        for (auto rho = 0; rho < nbasis; rho++) {
          for (auto nu = 0; nu < nbasis; nu++) {
            for (auto mu = 0; mu < nbasis; mu++) {
              E_k += w(k)*g_p(mu,nu,rho,sigma)*(2.0*g(mu,nu,rho,sigma) - g(mu,sigma,rho,nu));
            }
          }
        }
      }
      //std::cout << "E_k = " << -E_k/alpha << std::endl;
      E_MP2 += E_k;

    }

    E_MP2 = -E_MP2/alpha;

    //std::cout << "E_MP2 = " << E_MP2 << std::endl;

    const auto stop_n = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_elapsed_n = stop_n - start_n;

    if (n == 2) {
      printf(" Number of segments         LT_MP2        error/microHartree  time for n/sec\n");}
    printf("         %02d       %20.12f %20.12f   %10.5lf\n", n, E_MP2, (E_MP2 - E_MP2_can)*1e6, time_elapsed_n.count());
  //}
}

//Greens-Functions 2
void gf2(const int nbasis, const int ndocc, Eigen::VectorXd &E_orb, TensorRank4 &g) {

  std::cout << "Starting with GF procedure:" << std::endl;

  Eigen::MatrixXd F(nbasis,nbasis);
  F = Eigen::MatrixXd::Zero(nbasis,nbasis);
  for (auto i = 0; i < nbasis; i++) {
    for (auto j = 0; j < nbasis; j++) {
      if (i == j) {
        F(i,j) = E_orb(i);
      }
    }
  }

  bool SO = true;

  if (SO) {
    const int nso = 2*nbasis;
    const int ndocc_so = 2*ndocc;

    const int fc_so = 0;

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
    TensorRank4 t2_so(nso,nso,nso,nso);

    for (auto i = 0; i < ndocc_so; i++) {
      for (auto j = 0; j < ndocc_so; j++) {
        for (auto a = ndocc_so; a < nso; a++) {
          for (auto b = ndocc_so; b < nso; b++) {
            double dijab = Fso(i,i)+Fso(j,j)-Fso(a,a)-Fso(b,b);
            t2_so(i,a,j,b) = (g_so(i,a,j,b) - g_so(i,b,j,a))/dijab;
            E_MP2_SO += 0.25*(g_so(i,a,j,b) - g_so(i,b,j,a))*t2_so(i,a,j,b);
          }
        }
      }
    }

    int iter = 0;

    int k = 1;
    double sigma_kk = 0.0;
    double Ek_old = Fso(k,k);
    double Ek_new;
    double diff;

    bool diagonal = false;

    if (diagonal) {
      std::cout << "Diagonal" << std::endl;

      //diagonal GF2 method or D2
      diff = 1.0;
      while (fabs(diff) > 1e-10) {

        double sigma_kk_hpp = 0;
        double d_sigma_kk_hpp = 0;
        for (auto i = fc_so; i < ndocc_so; i++) {
          for (auto a = ndocc_so; a < nso; a++) {
            for (auto b = ndocc_so; b < nso; b++) {
              sigma_kk_hpp += 0.5*(g_so(k,a,i,b) - g_so(k,b,i,a)) * (g_so(a,k,b,i) - g_so(a,i,b,k))/
                  (Ek_old + Fso(i,i) - Fso(a,a) - Fso(b,b) );

              d_sigma_kk_hpp += 0.5*(g_so(k,a,i,b) - g_so(k,b,i,a)) * (g_so(a,k,b,i) - g_so(a,i,b,k))/
                  pow((Ek_old + Fso(i,i) - Fso(a,a) - Fso(b,b)),2);
            }
          }
        }

        double sigma_kk_hhp = 0;
        double d_sigma_kk_hhp = 0;
        for (auto i = fc_so; i < ndocc_so; i++) {
          for (auto j = fc_so; j < ndocc_so; j++) {
            for (auto a = ndocc_so; a < nso; a++) {
              sigma_kk_hhp += 0.5*(g_so(k,i,a,j) - g_so(k,j,a,i)) * (g_so(i,k,j,a) - g_so(i,a,j,k))/
                  (Ek_old + Fso(a,a) - Fso(i,i) - Fso(j,j));
              d_sigma_kk_hhp += 0.5*(g_so(k,i,a,j) - g_so(k,j,a,i)) * (g_so(i,k,j,a) - g_so(i,a,j,k))/
                  pow((Ek_old + Fso(a,a) - Fso(i,i) - Fso(j,j)),2);
            }
          }
        }


        Ek_new = Fso(k,k) + sigma_kk_hpp + sigma_kk_hhp;
        //double derivative = 1 + d_sigma_kk_hpp + d_sigma_kk_hhp;
        //Ek_new = Ek_old - (Ek_old - Fso(k,k) - sigma_kk_hpp - sigma_kk_hhp)/derivative;

        std::cout << Ek_new << std::endl;
        diff = Ek_new - Ek_old;
        Ek_old = Ek_new;
        iter++;
      }
    } else {

      //nondiagonal GF2 method
      diff = 1.0;
      while (fabs(diff) > 1e-10) {

        Eigen::MatrixXd Sigma(nso-fc_so,nso-fc_so);

        for (auto p = fc_so; p < nso; p++) {
          for (auto q = fc_so; q < nso; q++) {

            double sigma_kk_hpp = 0;
            for (auto i = fc_so; i < ndocc_so; i++) {
              for (auto a = ndocc_so; a < nso; a++) {
                for (auto b = ndocc_so; b < a; b++) {
                  sigma_kk_hpp += (g_so(p,a,i,b) - g_so(p,b,i,a)) * (g_so(a,q,b,i) - g_so(a,i,b,q))/
                      (Ek_old + Fso(i,i) - Fso(a,a) - Fso(b,b));
                }
              }
            }

            double sigma_kk_hhp = 0;
            for (auto i = fc_so; i < ndocc_so; i++) {
              for (auto j = fc_so; j < i; j++) {
                for (auto a = ndocc_so; a < nso; a++) {
                  sigma_kk_hhp += (g_so(p,i,a,j) - g_so(p,j,a,i)) * (g_so(i,q,j,a) - g_so(i,a,j,q))/
                      (Ek_old + Fso(a,a) - Fso(i,i) - Fso(j,j));
                }
              }
            }

            Sigma(p,q) = sigma_kk_hpp + sigma_kk_hhp;

          }
        }

        //std::cout << "Sigma = " << std::endl;
        //std::cout << Sigma << std::endl;

        Sigma += Fso;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma);
        Eigen::VectorXd Ep = es.eigenvalues();

        Ek_new = Ep(k);

        std::cout << Ek_new << std::endl;
        diff = Ek_new - Ek_old;
        Ek_old = Ek_new;
      }
    }

    std::cout << "D2: "<< std::endl;
    std::cout << "Fso(k,k) = " << Fso(k,k) << " a.u.   " << Fso(k,k) * 27.2114 << " eV "<< std::endl;
    std::cout << "Ek_new = " << Ek_new << " a.u.   " << Ek_new * 27.2114 << " eV "<< std::endl;
    printf("Ek_new %20.12f \n", Ek_new*27.21138602);


    double Ep_old = Fso(k,k);
    double Ep_new;

    //P3 method
    /*
    if (diagonal) {
      int p = 1;
      diff = 1.0;

      TensorRank4 W(nso,nso,nso,nso);
      for (auto i = fc_so; i < ndocc_so; i++) {
        for (auto j = fc_so; j < ndocc_so; j++) {
          for (auto a = ndocc_so; a < nso; a++) {
            W(p,a,i,j) = g_so(p,i,a,j) - g_so(p,j,a,i);
            double sum = 0.0;
            for (auto b = ndocc_so; b < nso; b++) {
              for (auto c = ndocc_so; c < nso; c++) {
                sum += 0.5*(g_so(p,b,a,c) - g_so(p,c,a,b))*(g_so(b,i,c,j) - g_so(b,j,c,i))/((Fso(i,i) + Fso(j,j) - Fso(b,b) - Fso(c,c)));
              }
            }
            W(p,a,i,j) += sum;

            double sum1 = 0.0;
            for (auto b = ndocc_so; b < nso; b++) {
              for (auto k = fc_so; k < ndocc_so; k++) {
                sum1 += (g_so(p,b,k,i) - g_so(p,i,k,b)) * (g_so(b,j,a,k) - g_so(b,k,a,j))/((Fso(k,k) + Fso(j,j) - Fso(b,b) - Fso(a,a)))
                      - (g_so(p,b,k,j) - g_so(p,j,k,b)) * (g_so(b,i,a,k) - g_so(b,k,a,i))/((Fso(k,k) + Fso(i,i) - Fso(b,b) - Fso(a,a)));
              }
            }
            W(p,a,i,j) += sum1;
          }
        }
      }

      std::cout << "starting with P3" << std::endl;
      while (fabs(diff) > 1e-10) {

        TensorRank4 U(nso,nso,nso,nso);
        for (auto i = fc_so; i < ndocc_so; i++) {
          for (auto j = fc_so; j < ndocc_so; j++) {
            for (auto a = ndocc_so; a < nso; a++) {

              double sum = 0.0;
              for (auto k = fc_so; k < ndocc_so; k++) {
                for (auto l = fc_so; l < ndocc_so; l++) {
                  sum += (g_so(p,k,a,l) - g_so(p,l,a,k)) * (g_so(k,i,l,j) - g_so(k,j,l,i))/((Ep_old + Fso(a,a) - Fso(k,k) - Fso(l,l)));
                }
              }
              U(p,a,i,j) = -0.5*sum;

              double sum1 = 0.0;
              for (auto b = ndocc_so; b < nso; b++) {
                for (auto k = fc_so; k < ndocc_so; k++) {
                  sum1 += (g_so(p,j,b,k) - g_so(p,k,b,j)) * (g_so(a,b,k,i) - g_so(a,i,k,b))/((Ep_old + Fso(b,b) - Fso(j,j) - Fso(k,k)))
                        - (g_so(p,i,b,k) - g_so(p,k,b,i)) * (g_so(a,b,k,j) - g_so(a,j,k,b))/((Ep_old + Fso(b,b) - Fso(i,i) - Fso(k,k)));
                }
              }
              U(p,a,i,j) += -sum1;
            }
          }
        }

        double sigma_hpp = 0.0;
        for (auto i = fc_so; i < ndocc_so; i++) {
          for (auto a = ndocc_so; a < nso; a++) {
            for (auto b = ndocc_so; b < nso; b++) {
              sigma_hpp += 0.5*(g_so(p,a,i,b) - g_so(p,b,i,a)) * (g_so(a,p,b,i) - g_so(a,i,b,p))/
                  (Ep_old + Fso(i,i) - Fso(a,a) - Fso(b,b));
            }
          }
        }

        double sigma_hhp = 0.0;
        for (auto i = fc_so; i < ndocc_so; i++) {
          for (auto j = fc_so; j < ndocc_so; j++) {
            for (auto a = ndocc_so; a < nso; a++) {
              sigma_hhp += 0.5*W(p,a,i,j)*(g_so(p,i,a,j) - g_so(p,j,a,i))/(Ep_old + Fso(a,a) - Fso(i,i) - Fso(j,j))
                         + 0.5*U(p,a,i,j)*(g_so(i,p,j,a) - g_so(i,a,j,p))/(Ep_old + Fso(a,a) - Fso(i,i) - Fso(j,j));
            }
          }
        }

        Ep_new = Fso(p,p) + sigma_hpp + sigma_hhp;
        std::cout << Ep_new << std::endl;
        diff = Ep_new - Ep_old;
        Ep_old = Ep_new;
      }

    } else {

      std::cout << "starting with W" << std::endl;
      TensorRank4 W(nso,nso,nso,nso);
      for (auto q = fc_so; q < nso; q++) {
        for (auto i = fc_so; i < ndocc_so; i++) {
          for (auto j = fc_so; j < ndocc_so; j++) {
            for (auto a = ndocc_so; a < nso; a++) {
              W(q,a,i,j) = g_so(q,i,a,j) - g_so(q,j,a,i);
              double sum = 0.0;
              for (auto b = ndocc_so; b < nso; b++) {
                for (auto c = ndocc_so; c < nso; c++) {
                  sum += 0.5*(g_so(q,b,a,c) - g_so(q,c,a,b))*(g_so(b,i,c,j) - g_so(b,j,c,i))/((Fso(i,i) + Fso(j,j) - Fso(b,b) - Fso(c,c)));
                }
              }
              W(q,a,i,j) += sum;

              double sum1 = 0.0;
              for (auto b = ndocc_so; b < nso; b++) {
                for (auto k = fc_so; k < ndocc_so; k++) {
                  sum1 += (g_so(q,b,k,i) - g_so(q,i,k,b)) * (g_so(b,j,a,k) - g_so(b,k,a,j))/((Fso(k,k) + Fso(j,j) - Fso(b,b) - Fso(a,a)))
                                          - (g_so(q,b,k,j) - g_so(q,j,k,b)) * (g_so(b,i,a,k) - g_so(b,k,a,i))/((Fso(k,k) + Fso(i,i) - Fso(b,b) - Fso(a,a)));
                }
              }
              W(q,a,i,j) += sum1;
            }
          }
        }
      }
      std::cout << "done with W" << std::endl;


      diff = 1.0;
      while (fabs(diff) > 1e-5) {

        std::cout << "starting with U" << std::endl;
        TensorRank4 U(nso,nso,nso,nso);
        for (auto p = fc_so; p < nso; p++) {
          for (auto i = fc_so; i < ndocc_so; i++) {
            for (auto j = fc_so; j < ndocc_so; j++) {
              for (auto a = ndocc_so; a < nso; a++) {

                double sum = 0.0;
                for (auto k = fc_so; k < ndocc_so; k++) {
                  for (auto l = fc_so; l < ndocc_so; l++) {
                    sum += (g_so(p,k,a,l) - g_so(p,l,a,k)) * (g_so(k,i,l,j) - g_so(k,j,l,i))/((Ep_old + Fso(a,a) - Fso(k,k) - Fso(l,l)));
                  }
                }
                U(p,a,i,j) = -0.5*sum;

                double sum1 = 0.0;
                for (auto b = ndocc_so; b < nso; b++) {
                  for (auto k = fc_so; k < ndocc_so; k++) {
                    sum1 += (g_so(p,j,b,k) - g_so(p,k,b,j)) * (g_so(a,b,k,i) - g_so(a,i,k,b))/((Ep_old + Fso(b,b) - Fso(j,j) - Fso(k,k)))
                                            - (g_so(p,i,b,k) - g_so(p,k,b,i)) * (g_so(a,b,k,j) - g_so(a,j,k,b))/((Ep_old + Fso(b,b) - Fso(i,i) - Fso(k,k)));
                  }
                }
                U(p,a,i,j) += -sum1;
              }
            }
          }
        }
        std::cout << "done with U" << std::endl;

        std::cout << "starting with Sigma" << std::endl;
        Eigen::MatrixXd Sigma(nso,nso);
        for (auto p = fc_so; p < nso; p++) {
          for (auto q = fc_so; q < nso; q++) {
            double sigma_hpp = 0;
            for (auto i = fc_so; i < ndocc_so; i++) {
              for (auto a = ndocc_so; a < nso; a++) {
                for (auto b = ndocc_so; b < nso; b++) {
                  sigma_hpp += 0.5*(g_so(p,a,i,b) - g_so(p,b,i,a)) * (g_so(a,q,b,i) - g_so(a,i,b,q))/
                      (Ep_old + Fso(i,i) - Fso(a,a) - Fso(b,b));
                }
              }
            }
            Sigma(p,q) = sigma_hpp;
            double sigma_hhp = 0;
            for (auto i = fc_so; i < ndocc_so; i++) {
              for (auto j = fc_so; j < ndocc_so; j++) {
                for (auto a = ndocc_so; a < nso; a++) {
                  sigma_hhp += 0.5*W(q,a,i,j)*(g_so(p,i,a,j) - g_so(p,j,a,i))/(Ep_old + Fso(a,a) - Fso(i,i) - Fso(j,j))
                                                 + 0.5*U(p,a,i,j)*(g_so(i,q,j,a) - g_so(i,a,j,q))/(Ep_old + Fso(a,a) - Fso(i,i) - Fso(j,j));
                }
              }
            }
            Sigma(p,q) += sigma_hhp;

            Sigma(p,q) += Fso(p,q);

          }
        }
        std::cout << "done with Sigma" << std::endl;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma);
        Eigen::VectorXd Ep = es.eigenvalues();

        Ep_new = Ep(1);

        std::cout << Ep_new << std::endl;
        diff = Ep_new - Ep_old;
        Ep_old = Ep_new;
      }

    }


    std::cout << "P3: "<< std::endl;
    std::cout << "Ep_new = " << Ep_new << " a.u.   " << Ep_new * 27.2114 << " eV "<< std::endl;
    */
    bool TDA;
    TDA = true;

    if (TDA) {

      std::cout << "TDA method:" << std::endl;

      int nvirt_so = nso-ndocc_so;

      std::cout << "number of all spin-orbitals: " << nso << std::endl;
      std::cout << "number of occupied spin-orbitals: " << ndocc_so << std::endl;
      std::cout << "number of unoccupied spin-orbitals: " << nvirt_so << std::endl;

      //defintion of sizes
      int h = ndocc_so;
      int p = nvirt_so;

      //H_hp_p2h
      int p2h = nvirt_so*ndocc_so*ndocc_so;
      std::cout << "p2h = " << p2h << std::endl;
      Eigen::MatrixXd H_hp_p2h(h+p,p2h);
      H_hp_p2h = Eigen::MatrixXd::Zero(h+p,p2h);

      int pp;
      for (auto hp = fc_so; hp < nso; hp++) {
        pp = 0;
        for (auto a = ndocc_so; a < nso; a++) {
          for (auto i = fc_so; i < ndocc_so; i++) {
            for (auto j = fc_so; j < ndocc_so; j++) {
              H_hp_p2h(hp,pp) = g_so(hp,i,a,j)-g_so(hp,j,a,i);
              pp++;
            }
          }
        }
      }
      std::cout << "H_hp_p2h = " << std::endl;
      //std::cout << H_hp_p2h << std::endl;

      Eigen::MatrixXd H_p2h_hp(p2h,h+p);
      H_p2h_hp = H_hp_p2h.transpose();
      std::cout << "H_p2h_hp = " << std::endl;


      //H_hp_h2p
      int h2p = ndocc_so*nvirt_so*nvirt_so;
      std::cout << "h2p = " << h2p << std::endl;
      Eigen::MatrixXd H_hp_h2p(h+p,h2p);
      H_hp_h2p = Eigen::MatrixXd::Zero(h+p,h2p);
      for (auto hp = fc_so; hp < nso; hp++) {
        pp = 0;
        for (auto i = fc_so; i < ndocc_so; i++) {
          for (auto a = ndocc_so; a < nso; a++) {
            for (auto b = ndocc_so; b < nso; b++) {

              H_hp_h2p(hp,pp) = g_so(hp,a,i,b) - g_so(hp,b,i,a);
              pp++;
            }
          }
        }
      }
      std::cout << "H_hp_h2p = " << std::endl;
      //std::cout << H_hp_h2p << std::endl;


      Eigen::MatrixXd H_h2p_hp(h2p,h+p);
      H_h2p_hp = H_hp_h2p.transpose();
      std::cout << "H_h2p_hp = " << std::endl;
      //std::cout << H_h2p_hp << std::endl;

      diff = 1;

      int orbital = 1;
      double E_old = Fso(orbital,orbital);
      double E_new;

      while (fabs(diff)>1e-05) {
        //H_p2h_p2h
        Eigen::MatrixXd H_p2h_p2h(p2h,p2h);
        H_p2h_p2h = Eigen::MatrixXd::Zero(p2h,p2h);

        pp = 0;
        for (auto a = ndocc_so; a < nso; a++) {
          for (auto i = fc_so; i < ndocc_so; i++) {
            for (auto j = fc_so; j < ndocc_so; j++) {

              int qq = 0;
              for (auto b = ndocc_so; b < nso; b++) {
                for (auto k = fc_so; k < ndocc_so; k++) {
                  for (auto l = fc_so; l < ndocc_so; l++) {

                    H_p2h_p2h(pp,qq) = E_old*((b == a)*(i == k)*(j == l)-(b == a)*(i == l)*(j == k))
                            - (Fso(a,a)*((b == a)*(i == l)*(j == k)-(b == a)*(i == k)*(j == l))
                              -Fso(i,i)*((b == a)*(i == l)*(j == k)-(b == a)*(i == k)*(j == l))
                              -Fso(j,j)*((b == a)*(i == l)*(j == k)-(b == a)*(i == k)*(j == l)));
                                       //-(a == b)*(g_so(i,k,j,l)-g_so(i,l,j,k)) + (l == j)*(g_so(b,a,i,k)-g_so(b,k,i,a))
                                       //-(l == i)*(g_so(b,a,j,k)-g_so(b,k,j,a)) - (j == k)*(g_so(b,a,i,l)-g_so(b,l,i,a))
                                       //+(k == i)*(g_so(b,a,j,l)-g_so(b,l,j,a)));

                    qq++;
                  }
                }
              }
              pp++;
            }
          }
        }
        std::cout << "H_p2h_p2h = " << std::endl;
        //std::cout << H_p2h_p2h << std::endl;

        std::cout << "Sigma = " << std::endl;
        Eigen::MatrixXd Sigma(h+p,h+p);
        Sigma = Eigen::MatrixXd::Zero(h+p,h+p);

        Eigen::MatrixXd x1 = H_p2h_p2h.ldlt().solve(H_p2h_hp);
        Sigma = H_hp_p2h*x1;

        //std::cout << Sigma << std::endl;

        Eigen::MatrixXd H_h2p_h2p(h2p,h2p);
        H_h2p_h2p = Eigen::MatrixXd::Zero(h2p,h2p);

        pp = 0;
        for (auto i = fc_so; i < ndocc_so; i++) {
          for (auto a = ndocc_so; a < nso; a++) {
          for (auto b = ndocc_so; b < nso; b++) {


              int qq = 0;
              for (auto j = fc_so; j < ndocc_so; j++) {
                for (auto c = ndocc_so; c < nso; c++) {
                for (auto d = ndocc_so; d < nso; d++) {

                    H_h2p_h2p(pp,qq) = E_old*((i == j)*(a == c)*(b == d)-(i == j)*(a == d)*(b == c))
                                    - ((i == j)*(a == c)*(b == d)*(Fso(a,a)+Fso(b,b)-Fso(i,i))
                                      -(i == j)*(a == d)*(b == c)*(Fso(a,a)+Fso(b,b)-Fso(i,i)));
                                       //+(i == j)*(g_so(a,c,b,d)-g_so(a,d,b,c)) - (c == a)*(g_so(b,d,j,i)-g_so(b,i,j,d))
                                       //+(d == a)*(g_so(b,c,j,i)-g_so(b,i,j,c)) + (c == b)*(g_so(a,d,j,i)-g_so(a,i,j,d))
                                       //-(d == b)*(g_so(a,c,j,i)-g_so(a,i,j,c)));
                    qq++;
                  }
                }
              }
              pp++;
            }
          }
        }
        std::cout << "H_h2p_h2p = " << std::endl;
        //std::cout << H_h2p_h2p << std::endl;

        Eigen::MatrixXd x2 = H_h2p_h2p.ldlt().solve(H_h2p_hp);
        Sigma += H_hp_h2p*x2;

        Sigma += Fso;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma);
        Eigen::VectorXd Ep = es.eigenvalues();

        E_new = Ep(orbital);

        //E_new = Fso(orbital,orbital) + Sigma(orbital,orbital);
        diff = E_new - E_old;
        std::cout << "diff = " << diff << "  E_new = " << E_new << std::endl;
        E_old = E_new;

      }
      std::cout << "h2p-TDA: "<< std::endl;
      std::cout << "E_new = " << E_new << " a.u.   " << E_new * 27.2114 << " eV "<< std::endl;

    }

    bool NR2_self;
    NR2_self = false;

    if (NR2_self) {
      std::cout << "NR2 self energy method:" << std::endl;

      int nvirt_so = nso-ndocc_so;

      std::cout << "number of all spin-orbitals: " << nso << std::endl;
      std::cout << "number of occupied spin-orbitals: " << ndocc_so << std::endl;
      std::cout << "number of unoccupied spin-orbitals: " << nvirt_so << std::endl;

      //definition of sizes
      const int h = ndocc_so-fc_so;
      const int p = nvirt_so;

      //H_h_p2h
      int p2h = p*h*(h-1)*0.5;
      std::cout << "p2h = " << p2h << std::endl;
      Eigen::MatrixXd H_h_p2h(h,p2h);
      H_h_p2h = Eigen::MatrixXd::Zero(h,p2h);

      int pp;
      for (auto i = fc_so; i < ndocc_so; i++) {
        pp = 0;
        for (auto a = ndocc_so; a < nso; a++) {
          for (auto k = fc_so; k < ndocc_so; k++) {
            for (auto j = fc_so; j < k; j++) {
              H_h_p2h(i-fc_so,pp) = g_so(i,j,a,k)-g_so(i,k,a,j);
              pp++;
            }
          }
        }
      }
      std::cout << "H_h_p2h = " << std::endl;
      //std::cout << H_h_p2h << std::endl;

      /*
      Eigen::MatrixXd H_p2h_h(p2h,h);
      H_p2h_h = H_h_p2h.transpose();
      std::cout << "H_p2h_h = " << std::endl;
      std::cout << H_p2h_h << std::endl;*/

      //H_p2h_h
      Eigen::MatrixXd H_p2h_h(p2h,h);
      H_p2h_h = Eigen::MatrixXd::Zero(p2h,h);

      pp = 0;
      for (auto a = ndocc_so; a < nso; a++) {
        for (auto k = fc_so; k < ndocc_so; k++) {
          for (auto j = fc_so; j < k; j++) {

            for (auto i = fc_so; i < ndocc_so; i++) {

              double sum = 0.0;
              for (auto b = ndocc_so; b < nso; b++) {
                for (auto c = ndocc_so; c < nso; c++) {
                  sum -= 0.5*(g_so(b,i,c,a)-g_so(b,a,c,i))*(g_so(b,j,c,k)-g_so(b,k,c,j))/(Fso(b,b)+Fso(c,c)-Fso(j,j)-Fso(k,k));
                }
              }

              H_p2h_h(pp,i-fc_so) += sum;

              sum = 0.0;
              for (auto b = ndocc_so; b < nso; b++) {
                for (auto l = fc_so; l < ndocc_so; l++) {
                  sum -= (g_so(b,i,j,l)-g_so(b,l,j,i))*(g_so(b,k,a,l)-g_so(b,l,a,k))/(Fso(b,b)+Fso(a,a)-Fso(k,k)-Fso(l,l))
                                    -(g_so(b,i,k,l)-g_so(b,l,k,i))*(g_so(b,j,a,l)-g_so(b,l,a,j))/(Fso(b,b)+Fso(a,a)-Fso(j,j)-Fso(l,l));
                }
              }
              H_p2h_h(pp,i-fc_so) += sum;

            }
            pp++;
          }
        }
      }
      std::cout << "H_p2h_h = " << std::endl;
      //std::cout << H_p2h_h << std::endl;


      //H_p_p2h
      Eigen::MatrixXd H_p_p2h(p,p2h);
      H_p_p2h = Eigen::MatrixXd::Zero(p,p2h);
      for (auto a = ndocc_so; a < nso; a++) {
        pp = 0;
        for (auto b = ndocc_so; b < nso; b++) {
          for (auto j = fc_so; j < ndocc_so; j++) {
            for (auto i = fc_so; i < j; i++) {

              H_p_p2h(a-ndocc_so,pp) = g_so(a,i,b,j) - g_so(a,j,b,i);
              pp++;
            }
          }
        }
      }
      std::cout << "H_p_p2h = " << std::endl;
      //std::cout << H_p_p2h << std::endl;

      Eigen::MatrixXd H_p2h_p(p2h,p);
      H_p2h_p = H_p_p2h.transpose();
      std::cout << "H_p2h_p = " << std::endl;
      //std::cout << H_p2h_p << std::endl;

      //H_h_h2p
      int h2p = h*p*(p-1)*0.5;
      std::cout << "h2p = " << h2p << std::endl;
      Eigen::MatrixXd H_h_h2p(h,h2p);
      H_h_h2p = Eigen::MatrixXd::Zero(h,h2p);
      for (auto i = fc_so; i < ndocc_so; i++) {
        pp = 0;
        for (auto j = fc_so; j < ndocc_so; j++) {
          for (auto b = ndocc_so; b < nso; b++) {
            for (auto a = ndocc_so; a < b; a++) {
              H_h_h2p(i-fc_so,pp) = g_so(i,a,j,b) - g_so(i,b,j,a);
              pp++;
            }
          }
        }
      }
      std::cout << "H_h_h2p = " << std::endl;
      //std::cout << H_h_h2p << std::endl;

      Eigen::MatrixXd H_h2p_h(h2p,h);
      H_h2p_h = H_h_h2p.transpose();
      std::cout << "H_h2p_h = " << std::endl;
      //std::cout << H_h2p_h << std::endl;

      Eigen::MatrixXd H_p_h2p(p,h2p);
      H_p_h2p = Eigen::MatrixXd::Zero(p,h2p);
      for (auto a = ndocc_so; a < nso; a++) {
        pp = 0;
        for (auto j = fc_so; j < ndocc_so; j++) {
          for (auto c = ndocc_so; c < nso; c++) {
            for (auto b = ndocc_so; b < c; b++) {
              H_p_h2p(a-ndocc_so,pp) = g_so(a,b,j,c) - g_so(a,c,j,b);
              pp++;
            }
          }
        }
      }
      std::cout << "H_p_h2p = " << std::endl;
      //std::cout << H_p_h2p << std::endl;

      Eigen::MatrixXd H_h2p_p(h2p,p);
      H_h2p_p = H_p_h2p.transpose();
      std::cout << "H_h2p_p = " << std::endl;
      //std::cout << H_h2p_p << std::endl;

      diff = 1;
      int orbital = 6-fc_so;
      double E_old = Fso(orbital,orbital);
      double E_new;

      while (fabs(diff)>1e-10) {
        //H_p2h_p2h
        Eigen::MatrixXd H_p2h_p2h(p2h,p2h);
        H_p2h_p2h = Eigen::MatrixXd::Zero(p2h,p2h);

        pp = 0;
        for (auto a = ndocc_so; a < nso; a++) {
          for (auto i = fc_so; i < ndocc_so; i++) {
            for (auto j = fc_so; j < i; j++) {

              int qq = 0;
              for (auto b = ndocc_so; b < nso; b++) {
                for (auto k = fc_so; k < ndocc_so; k++) {
                  for (auto l = fc_so; l < k; l++) {

                    H_p2h_p2h(pp,qq) = E_old*(a == b)*(i == k)*(j == l) - ((a == b)*(i == k)*(j == l)*(Fso(i,i)+Fso(j,j)-Fso(a,a))
                        -(a == b)*(g_so(i,k,j,l)-g_so(i,l,j,k)) + (l == j)*(g_so(b,a,i,k)-g_so(b,k,i,a))
                        -(l == i)*(g_so(b,a,j,k)-g_so(b,k,j,a))
                        - (j == k)*(g_so(b,a,i,l)-g_so(b,l,i,a))
                        +(k == i)*(g_so(b,a,j,l)-g_so(b,l,j,a)));

                    qq++;
                  }
                }
              }
              pp++;
            }
          }
        }
        std::cout << "H_p2h_p2h = " << std::endl;
        //std::cout << H_p2h_p2h << std::endl;

        Eigen::MatrixXd x1 = H_p2h_p2h.ldlt().solve(H_h_p2h.transpose()+0.5*H_p2h_h);

        Eigen::MatrixXd Sigma_ij(h,h);
        Sigma_ij = Eigen::MatrixXd::Zero(h,h);
        Sigma_ij = (H_h_p2h+0.5*H_p2h_h.transpose())*x1;

        std::cout << "Sigma_ij = " << std::endl;
        //std::cout << Sigma_ij << std::endl;

        Eigen::MatrixXd Sigma_aj(p,h);
        Sigma_aj = Eigen::MatrixXd::Zero(p,h);

        Sigma_aj = H_p_p2h*x1;

        std::cout << "Sigma_aj = " << std::endl;
        //std::cout << Sigma_aj << std::endl;

        Eigen::MatrixXd Sigma_ia(h,p);
        Sigma_ia = Eigen::MatrixXd::Zero(h,p);
        Sigma_ia = Sigma_aj.transpose();

        std::cout << "Sigma_ia = " << std::endl;
        //std::cout << Sigma_ia << std::endl;

        Eigen::MatrixXd x2 = H_p2h_p2h.ldlt().solve(H_p2h_p);

        Eigen::MatrixXd Sigma_ab(p,p);
        Sigma_ab = Eigen::MatrixXd::Zero(p,p);
        Sigma_ab = H_p_p2h*x2;

        std::cout << "Sigma_ab = " << std::endl;
        //std::cout << Sigma_ab << std::endl;


        Eigen::MatrixXd Sigma(h+p,h+p);
        Sigma = Eigen::MatrixXd::Zero(h+p,h+p);


        for (int pp = 0; pp < h+p; pp++) {
          for (int qq = 0; qq < h+p; qq++) {

            //H_h_h block
            if (pp < h && qq < h) {
              Sigma(pp,qq) = Sigma_ij(pp,qq);
            }

            //H_h_p block
            if (pp < h && qq >= h) {
              Sigma(pp,qq) = Sigma_ia(pp,qq-h);
            }

            //H_p_h block
            if (pp >= h && qq < h) {
              Sigma(pp,qq) = Sigma_aj(pp-h,qq);
            }

            //H_p_p block
            if (pp >= h && qq >= h) {
              Sigma(pp,qq) = Sigma_ab(pp-h,qq-h);
            }

          }
        }

        std::cout << "Sigma = " << std::endl;
        //std::cout << Sigma << std::endl;

        for (auto pp = fc_so; pp < nso; pp++) {
          for (auto qq = fc_so; qq < nso; qq++) {

            double sigma_kk_hpp = 0;
            for (auto i = fc_so; i < ndocc_so; i++) {
              for (auto a = ndocc_so; a < nso; a++) {
                for (auto b = ndocc_so; b < a; b++) {
                  sigma_kk_hpp += (g_so(pp,a,i,b) - g_so(pp,b,i,a)) * (g_so(a,qq,b,i) - g_so(a,i,b,qq))/
                      (E_old + Fso(i,i) - Fso(a,a) - Fso(b,b));
                }
              }
            }

            Sigma(pp-fc_so,qq-fc_so) += sigma_kk_hpp + Fso(pp,qq);

          }
        }

        //std::cout << Sigma << std::endl;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma);
        Eigen::VectorXd Ep = es.eigenvalues();

        E_new = Ep(orbital);

        diff = E_new - E_old;
        std::cout << "diff = " << diff << "  E_new = " << E_new << std::endl;
        E_old = E_new;

      }
      std::cout << "NR2: "<< std::endl;
      std::cout << "E_new = " << E_new << " a.u.   " << E_new * 27.21138602 << " eV "<< std::endl;
      printf("Ek_new %20.12f \n", E_new*27.21138602);
    }


    bool NR2;
    NR2 = false;

    if (NR2) {
      std::cout << "NR2 method:" << std::endl;

      int nvirt_so = nso-ndocc_so;

      std::cout << "number of all spin-orbitals: " << nso << std::endl;
      std::cout << "number of occupied spin-orbitals: " << ndocc_so << std::endl;
      std::cout << "number of unoccupied spin-orbitals: " << nvirt_so << std::endl;

      //defintion of sizes
      int h = ndocc_so;
      int p = nvirt_so;

      //construction of block of total Hamiltonian

      //H_h_h block
      Eigen::MatrixXd H_h_h(h,h);
      H_h_h = Eigen::MatrixXd::Zero(h,h);

      for (auto i = fc_so; i < h; i++) {
        for (auto j = fc_so; j < h; j++) {
          H_h_h(i,j) = Fso(i,j);
        }
      }
      std::cout << "H_h_h = " << std::endl;
      //std::cout << H_h_h << std::endl;

      //H_h_p block
      Eigen::MatrixXd H_h_p(h,p);
      H_h_p = Eigen::MatrixXd::Zero(h,p);
      std::cout << "H_h_p = " << std::endl;
      //std::cout << H_h_p << std::endl;

      //H_p_h block
      Eigen::MatrixXd H_p_h(p,h);
      H_p_h = Eigen::MatrixXd::Zero(nvirt_so,ndocc_so);
      std::cout << "H_p_h = " << std::endl;
      //std::cout << H_p_h << std::endl;

      //H_p_p block
      Eigen::MatrixXd H_p_p(nvirt_so,nvirt_so);
      H_p_p = Eigen::MatrixXd::Zero(nvirt_so,nvirt_so);

      for (auto a = ndocc_so; a < nso; a++) {
        for (auto b = ndocc_so; b < nso; b++) {
          H_p_p(a-ndocc_so,b-ndocc_so) = Fso(a,b);
        }
      }
      std::cout << "H_p_p = " << std::endl;
      //std::cout << H_p_p << std::endl;

      //H_h_2hp
      int p2h = nvirt_so*ndocc_so*(ndocc_so-1)*0.5;
      std::cout << "p2h = " << p2h << std::endl;
      Eigen::MatrixXd H_h_2hp(h,p2h);
      H_h_2hp = Eigen::MatrixXd::Zero(h,p2h);

      int pp;
      for (auto i = fc_so; i < h; i++) {
        pp = 0;
        for (auto a = ndocc_so; a < nso; a++) {
          for (auto k = fc_so; k < ndocc_so; k++) {
            for (auto j = fc_so; j < k; j++) {
              H_h_2hp(i,pp) = g_so(i,j,a,k)-g_so(i,k,a,j);
              pp++;
            }
          }
        }
      }
      std::cout << "H_h_2hp = " << std::endl;
      //std::cout << H_h_2hp << std::endl;

      //H_p_p2h
      Eigen::MatrixXd H_p_2hp(p,p2h);
      H_p_2hp = Eigen::MatrixXd::Zero(p,p2h);
      for (auto a = ndocc_so; a < nso; a++) {
        pp = 0;
        for (auto b = ndocc_so; b < nso; b++) {
          for (auto j = fc_so; j < ndocc_so; j++) {
            for (auto i = fc_so; i < j; i++) {

              H_p_2hp(a-ndocc_so,pp) = g_so(a,i,b,j) - g_so(a,j,b,i);
              pp++;
            }
          }
        }
      }
      std::cout << "H_p_2hp = " << std::endl;
      //std::cout << H_p_2hp << std::endl;

      Eigen::MatrixXd H_2hp_p(p2h,p);
      H_2hp_p = H_p_2hp.transpose();
      std::cout << "H_2hp_p = " << std::endl;
      //std::cout << H_2hp_p << std::endl;

      //H_2hp_h
      Eigen::MatrixXd H_2hp_h(p2h,h);
      H_2hp_h = Eigen::MatrixXd::Zero(p2h,h);

      pp = 0;
      for (auto a = ndocc_so; a < nso; a++) {
      for (auto k = fc_so; k < ndocc_so; k++) {
        for (auto j = fc_so; j < k; j++) {


            for (auto i = fc_so; i < h; i++) {
              H_2hp_h(pp,i) = g_so(j,i,k,a)-g_so(j,a,k,i);

              double sum = 0.0;
              for (auto b = ndocc_so; b < nso; b++) {
                for (auto c = ndocc_so; c < nso; c++) {
                  sum -= 0.5*(g_so(b,i,c,a)-g_so(b,a,c,i))*(g_so(b,j,c,k)-g_so(b,k,c,j))/(Fso(b,b)+Fso(c,c)-Fso(j,j)-Fso(k,k));
                }
              }

              H_2hp_h(pp,i) += sum;

              sum = 0.0;
              for (auto b = ndocc_so; b < nso; b++) {
                for (auto l = fc_so; l < ndocc_so; l++) {
                  sum -= (g_so(b,i,j,l)-g_so(b,l,j,i))*(g_so(b,k,a,l)-g_so(b,l,a,k))/(Fso(b,b)+Fso(a,a)-Fso(k,k)-Fso(l,l))
                          -(g_so(b,i,k,l)-g_so(b,l,k,i))*(g_so(b,j,a,l)-g_so(b,l,a,j))/(Fso(b,b)+Fso(a,a)-Fso(j,j)-Fso(l,l));
                }
              }
              H_2hp_h(pp,i) += sum;

            }
            pp++;
          }
        }
      }
      std::cout << "H_2hp_h = " << std::endl;
      //std::cout << H_2hp_h << std::endl;

      //H_2hp_2hp
      Eigen::MatrixXd H_2hp_2hp(p2h,p2h);
      H_2hp_2hp = Eigen::MatrixXd::Zero(p2h,p2h);

      pp = 0;
      for (auto a = ndocc_so; a < nso; a++) {
        for (auto j = fc_so; j < ndocc_so; j++) {
          for (auto i = fc_so; i < j; i++) {

            int qq = 0;
            for (auto b = ndocc_so; b < nso; b++) {
              for (auto l = fc_so; l < ndocc_so; l++) {
                for (auto k = fc_so; k < l; k++) {
                  H_2hp_2hp(pp,qq) = (a == b)*(i == k)*(j == l)*(Fso(i,i)+Fso(j,j)-Fso(a,a));

                  H_2hp_2hp(pp,qq) += -(a == b)*(g_so(i,k,j,l)-g_so(i,l,j,k));

                  H_2hp_2hp(pp,qq) += (i == k)*(g_so(b,a,j,l)-g_so(b,l,j,a)) - (i == l)*(g_so(b,a,j,k)-g_so(b,k,j,a))
                                     -(j == k)*(g_so(b,a,i,l)-g_so(b,l,i,a)) + (j == l)*(g_so(b,a,i,k)-g_so(b,k,i,a));

                  qq++;
                }
              }
            }

            pp++;
          }
        }
      }
      std::cout << "H_2hp_2hp = " << std::endl;
      //std::cout << H_2hp_2hp << std::endl;


      //H_h_2ph
      int h2p = ndocc_so*nvirt_so*(nvirt_so-1)*0.5;
      std::cout << "h2p = " << h2p << std::endl;
      Eigen::MatrixXd H_h_2ph(h,h2p);
      H_h_2ph = Eigen::MatrixXd::Zero(h,h2p);
      for (auto i = fc_so; i < ndocc_so; i++) {
        pp = 0;
        for (auto j = fc_so; j < ndocc_so; j++) {
          for (auto b = ndocc_so; b < nso; b++) {
            for (auto a = ndocc_so; a < b; a++) {
              H_h_2ph(i,pp) = g_so(i,a,j,b) - g_so(i,b,j,a);
              pp++;
            }
          }
        }
      }
      std::cout << "H_h_2ph = " << std::endl;
      //std::cout << H_h_2ph << std::endl;


      Eigen::MatrixXd H_2ph_h(h2p,h);
      H_2ph_h = H_h_2ph.transpose();
      std::cout << "H_2ph_h = " << std::endl;
      //std::cout << H_2ph_h << std::endl;

      Eigen::MatrixXd H_p_2ph(p,h2p);
      H_p_2ph = Eigen::MatrixXd::Zero(p,h2p);
      for (auto a = ndocc_so; a < nso; a++) {
        pp = 0;
        for (auto j = fc_so; j < ndocc_so; j++) {
          for (auto c = ndocc_so; c < nso; c++) {
            for (auto b = ndocc_so; b < c; b++) {
              H_p_2ph(a-ndocc_so,pp) = g_so(a,b,j,c) - g_so(a,c,j,b);
              pp++;
            }
          }
        }
      }
      //std::cout << pp << std::endl;
      std::cout << "H_p_2ph = " << std::endl;
      //std::cout << H_p_2ph << std::endl;

      Eigen::MatrixXd H_2ph_p(h2p,h);
      H_2ph_p = H_p_2ph.transpose();
      std::cout << "H_2ph_p = " << std::endl;
      //std::cout << H_2ph_p << std::endl;

      Eigen::MatrixXd H_2ph_2ph(h2p,h2p);
      H_2ph_2ph = Eigen::MatrixXd::Zero(h2p,h2p);

      pp = 0;
      for (auto i = fc_so; i < ndocc_so; i++) {
        for (auto b = ndocc_so; b < nso; b++) {
          for (auto a = ndocc_so; a < b; a++) {

            int qq = 0;
            for (auto j = fc_so; j < ndocc_so; j++) {
              for (auto d = ndocc_so; d < nso; d++) {
                for (auto c = ndocc_so; c < d; c++) {
                  H_2ph_2ph(pp,qq) = (i == j)*(a == c)*(b == d)*(Fso(a,a)+Fso(b,b)-Fso(i,i));
                  qq++;
                }
              }
            }
            pp++;
          }
        }
      }
      std::cout << "H_2ph_2ph = " << std::endl;
      //std::cout << H_2ph_2ph << std::endl;


      //construction of the total Hamiltonian
      int n_tot = h + p + p2h + h2p;
      Eigen::MatrixXd H(n_tot,n_tot);
      H = Eigen::MatrixXd::Zero(n_tot,n_tot);

      for (int pp = 0; pp < n_tot; pp++) {
        for (int qq = 0; qq < n_tot; qq++) {

          //H_h_h block
          if (pp < h && qq < h) {
            H(pp,qq) = H_h_h(pp,qq);
          }

          //H_h_p block
          if (pp < h && qq >= h) {
            if ((qq < n_tot - p2h-h2p)) {
              H(pp,qq) = H_h_p(pp,qq-ndocc_so);
            }
          }

          //H_p_h block
          if (pp >= h && qq < h) {
            if (pp < n_tot - p2h-h2p) {
              H(pp,qq) = H_p_h(pp-ndocc_so,qq);
            }
          }

          //H_p_p block
          if (pp >= h && qq >= h) {
            if ((pp < n_tot - p2h-h2p) && (qq < n_tot - p2h-h2p)) {
              H(pp,qq) = H_p_p(pp-ndocc_so,qq-ndocc_so);
            }
          }

          //H_h_2hp block
          if (pp < h && qq >= h+p) {
            if (qq < n_tot - h2p) {
              H(pp,qq) = H_h_2hp(pp,qq-h-p);
            }
          }

          //H_2hp_h block
          if (pp >= h+p && qq < h) {
            if (pp < n_tot - h2p) {
              H(pp,qq) = H_2hp_h(pp -h-p,qq);
            }
          }

          //H_p_2hp block
          if (pp >= h && qq >= h+p) {
            if (pp < h+p && qq < n_tot - h2p) {
              H(pp,qq) = H_p_2hp(pp-h,qq-h-p);
            }
          }

          //H_2hp_p block
          if (pp >= h+p && qq >= h) {
            if (qq < h+p && pp < n_tot - h2p) {
              H(pp,qq) = H_2hp_p(pp-h-p,qq-h);
            }
          }

          //H_2hp_2hp block
          if (pp >= h+p && qq >= h+p) {
            if (pp < n_tot - h2p && qq < n_tot - h2p) {
              H(pp,qq) = H_2hp_2hp(pp-h-p,qq-h-p);
            }
          }

          //H_h_2ph block
          if (pp < h && qq >= h+p+p2h) {
            H(pp,qq) = H_h_2ph(pp,qq-h-p-p2h);
          }


          //H_2ph_h block
          if (pp >= h+p+p2h && qq < h) {
            H(pp,qq) = H_2ph_h(pp-h-p-p2h,qq);
          }

          //H_p_2ph block
          if (pp >= h && qq >= h+p+p2h) {
            if (pp < h+p) {
              H(pp,qq) = H_p_2ph(pp-h,qq-h-p-p2h);
            }
          }

          //H_2ph_p block
          if (pp >= h+p+p2h && qq >= h) {
            if (qq < h+p) {
              H(pp,qq) = H_2ph_p(pp-h-p-p2h,qq-h);
            }
          }

          //H_2ph_2ph block
          if (pp >= h+p+p2h && qq >= h+p+p2h) {
            H(pp,qq) = H_2ph_2ph(pp-h-p-p2h,qq-h-p-p2h);
          }

        }
      }
      std::cout << "H = " << std::endl;
      //std::cout << H << std::endl;

      Eigen::MatrixXd H_tot(n_tot,n_tot);
      H_tot = 0.5*(H + H.transpose());

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> self(H_tot);
      Eigen::VectorXd E = self.eigenvalues();
      Eigen::MatrixXd U = self.eigenvectors();
      std::cout << E << std::endl;
      printf("E %20.12f \n", E(7));
      /*
    //inverse iteration method
    Eigen::VectorXd v(n_tot);
    for (auto p = 0; p < n_tot; p++) {
      v(p) = 1.0;
    }
    double norm = v.transpose()*v;
    v = v/sqrt(norm);

    Eigen::MatrixXd Imu(n_tot,n_tot);
    for (auto p = 0; p < n_tot; p++) {
      for (auto q = 0; q < n_tot; q++) {
        if (p == q) {
          Imu(p,q) = Ep_new;
        }
      }
    }


    diff = 1;
    while (fabs(diff) > 1e-04) {
      double lamda1 =v.transpose()*H_tot*v;
      Eigen::VectorXd w(n_tot);
      w = (H_tot - Imu).householderQr().solve(v);
      double norm = sqrt(w.transpose()*w);
      v = w/norm;
      double lamda2 =v.transpose()*H_tot*v;
      diff = lamda2 - lamda1;
      std::cout << "lamda = " << lamda2 << std::endl;
    }
    double lamda = v.transpose()*H_tot*v;

    std::cout << lamda*27.2114 << std::endl;*/
    }


  } else {

    const int fc = 0;
    int k = 4;
    double sigma_kk = 0.0;
    double Ek_old = F(k,k);
    double Ek_new;
    double diff = 1.0;

    bool diagonal = true;
    if (diagonal) {
      //while (fabs(diff) > 1e-10) {

      std::cout << g(0,0,0,0) << std::endl;
        double sigma_kk_hpp = 0;
        for (auto i = fc; i < ndocc; i++) {
          double sigma_i =0.0;
          for (auto a = ndocc; a < nbasis; a++) {
            for (auto b = ndocc; b < nbasis; b++) {
              sigma_kk_hpp += (g(k,a,i,b)) * (2.0*g(a,k,b,i) - g(a,i,b,k))/
                  (Ek_old + F(i,i) - F(a,a) - F(b,b));
                  // (g(k,a,i,b)) * (2.0*g(a,k,b,i) - g(a,i,b,k))/
                  //(Ek_old + F(i,i) - F(a,a) - F(b,b)
                   //-(0.5*(2.0*g(a,a,b,b) - g(a,b,b,a))
                   //     -0.5*(2.0*g(a,a,i,i) - g(a,i,i,a))
                   //     -0.5*(2.0*g(b,b,i,i) - g(b,i,i,b))));
            }
          }
          //std::cout << "sigma_i = " << sigma_i << std::endl;
        }
        std::cout << "sigma_kk_hpp = " << sigma_kk_hpp << std::endl;

        double sigma_kk_hhp = 0;
        for (auto i = fc; i < ndocc; i++) {
          for (auto j = fc; j < ndocc; j++) {
            for (auto a = ndocc; a < nbasis; a++) {
              sigma_kk_hhp += (g(k,i,a,j)) * (2.0*g(i,k,j,a) - g(i,a,j,k))/
                  (Ek_old + F(a,a) - F(i,i) - F(j,j) );
            }
          }
        }

        std::cout << "sigma_kk_hhp = " << sigma_kk_hhp << std::endl;
        Ek_new = F(k,k) + sigma_kk_hpp + sigma_kk_hhp;

        std::cout << Ek_new << std::endl;
        diff = Ek_new - Ek_old;
        Ek_old = Ek_new;
      //}
    } else {

      //nondiagonal GF2 method
      diff = 1.0;
      //while (fabs(diff) > 1e-10) {

        Eigen::MatrixXd Sigma(nbasis-fc,nbasis-fc);

        for (auto p = fc; p < nbasis; p++) {
          for (auto q = fc; q < nbasis; q++) {

            double sigma_kk_hpp = 0;
            for (auto i = fc; i < ndocc; i++) {
              for (auto a = ndocc; a < nbasis; a++) {
                for (auto b = ndocc; b < nbasis; b++) {
                  sigma_kk_hpp += (g(p,a,i,b)) * (2.0*g(a,q,b,i) - g(a,i,b,q))/
                      (Ek_old + F(i,i) - F(a,a) - F(b,b));
                }
              }
            }

            double sigma_kk_hhp = 0;
            for (auto i = fc; i < ndocc; i++) {
              for (auto j = fc; j < ndocc; j++) {
                for (auto a = ndocc; a < nbasis; a++) {
                  sigma_kk_hhp += (g(p,i,a,j)) * (2.0*g(i,q,j,a) - g(i,a,j,q))/
                     (Ek_old + F(a,a) - F(i,i) - F(j,j));
                }
              }
            }

            Sigma(p-fc,q-fc) = sigma_kk_hpp + sigma_kk_hhp;

          }
        }

        std::cout << "Sigma = " << std::endl;
        std::cout << Sigma << std::endl;

        //Sigma += F;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma);
        Eigen::VectorXd Ep = es.eigenvalues();

        Ek_new = Ep(k-fc);

        std::cout << Ek_new << std::endl;
        diff = Ek_new - Ek_old;
        Ek_old = Ek_new;
      //}
    }

    std::cout << "D2: "<< std::endl;
    std::cout << "F(k,k) = " << F(k,k) << " a.u.   " << F(k,k) * 27.2114 << " eV "<< std::endl;
    std::cout << "Ek_new = " << Ek_new << " a.u.   " << Ek_new * 27.2114 << " eV "<< std::endl;
    printf("Ek_new %20.12f \n", Ek_new*27.21138602);


    /*
    double Ep_old = F(3,3);
    double Ep_new;

    int p = 3;
    diff = 1.0;

    TensorRank4 W(nbasis,nbasis,nbasis,nbasis);
    for (auto i = fc; i < ndocc; i++) {
      for (auto j = fc; j < ndocc; j++) {
        for (auto a = ndocc; a < nbasis; a++) {
          W(p,a,i,j) = 2.0*g(p,i,a,j) - g(p,j,a,i);
          double sum = 0.0;
          for (auto b = ndocc; b < nbasis; b++) {
            for (auto c = ndocc; c < nbasis; c++) {
              sum += (g(p,b,a,c))*(2.0*g(b,i,c,j) - g(b,j,c,i))/((F(i,i) + F(j,j) - F(b,b) - F(c,c)));
            }
          }
          W(p,a,i,j) += sum;

          double sum1 = 0.0;
          for (auto b = ndocc; b < nbasis; b++) {
            for (auto k = fc; k < ndocc; k++) {
              sum1 += (1.0*g(p,b,k,i)*g(b,j,a,k) - 2.0*g(p,b,k,i)*g(b,k,a,j)
                            -2.0*g(p,i,k,b)*g(b,j,a,k) + 4.0*g(p,i,k,b)*g(b,k,a,j))
                            /((F(k,k) + F(j,j) - F(b,b) - F(a,a)))

                            +(-2.0*g(p,b,k,j)*g(b,i,a,k) + 1.0*g(p,b,k,j)*g(b,k,a,i)
                            +1.0*g(p,j,k,b)*g(b,i,a,k) - 2.0*g(p,j,k,b)*g(b,k,a,i))
                                /((F(k,k) + F(i,i) - F(b,b) - F(a,a)))

                            +(-2.0*g(p,b,k,j)*g(b,i,a,k) + 1.0*g(p,b,k,j)*g(b,k,a,i)
                            +1.0*g(p,j,k,b)*g(b,i,a,k) - 2.0*g(p,j,k,b)*g(b,k,a,i))
                            /((F(k,k) + F(i,i) - F(b,b) - F(a,a)))

                            +(+1.0*g(p,b,k,i)*g(b,j,a,k) - 2.0*g(p,b,k,i)*g(b,k,a,j)
                            -2.0*g(p,i,k,b)*g(b,j,a,k) + 4.0*g(p,i,k,b)*g(b,k,a,j))
                                /((F(k,k) + F(j,j) - F(b,b) - F(a,a)));
            }
          }
          W(p,a,i,j) += 0.5*sum1;
        }
      }
    }

    while (fabs(diff) > 1e-10) {

      TensorRank4 U(nbasis,nbasis,nbasis,nbasis);
      for (auto i = fc; i < ndocc; i++) {
        for (auto j = fc; j < ndocc; j++) {
          for (auto a = ndocc; a < nbasis; a++) {

            double sum = 0.0;
            for (auto k = fc; k < ndocc; k++) {
              for (auto l = fc; l < ndocc; l++) {
                sum += (g(p,k,a,l)) * (2.0*g(k,i,l,j) - g(k,j,l,i))/((Ep_old + F(a,a) - F(k,k) - F(l,l)));
              }
            }
            U(p,a,i,j) = -sum;

            double sum1 = 0.0;
            for (auto b = ndocc; b < nbasis; b++) {
              for (auto k = fc; k < ndocc; k++) {
                sum1 += ((+1.0*g(p,j,b,k)*g(a,b,k,i) - 2.0*g(p,j,b,k)*g(a,i,k,b)
                    -2.0*g(p,k,b,j)*g(a,b,k,i) + 1.0*g(p,k,b,j)*g(a,i,k,b))
                    /((Ep_old + F(b,b) - F(j,j) - F(k,k)))

                  +(-2.0*g(p,i,b,k)*g(a,b,k,j) + 4.0*g(p,i,b,k)*g(a,j,k,b)
                    +1.0*g(p,k,b,i)*g(a,b,k,j) - 2.0*g(p,k,b,i)*g(a,j,k,b))
                    /((Ep_old + F(b,b) - F(i,i) - F(k,k)))

                  +(-2.0*g(p,i,b,k)*g(a,b,k,j) + 4.0*g(p,i,b,k)*g(a,j,k,b)
                    +1.0*g(p,k,b,i)*g(a,b,k,j) - 2.0*g(p,k,b,i)*g(a,j,k,b))
                    /((Ep_old + F(b,b) - F(i,i) - F(k,k)))

                  +(+1.0*g(p,j,b,k)*g(a,b,k,i) - 2.0*g(p,j,b,k)*g(a,i,k,b)
                    -2.0*g(p,k,b,j)*g(a,b,k,i) + 1.0*g(p,k,b,j)*g(a,i,k,b))
                    /((Ep_old + F(b,b) - F(j,j) - F(k,k))));
              }
            }
            U(p,a,i,j) += -0.5*sum1;
          }
        }
      }

      double sigma_hpp = 0.0;
      for (auto i = fc; i < ndocc; i++) {
        for (auto a = ndocc; a < nbasis; a++) {
          for (auto b = ndocc; b < nbasis; b++) {
            sigma_hpp += (g(p,a,i,b)) * (2.0*g(a,p,b,i) - g(a,i,b,p))/
                (Ep_old + F(i,i) - F(a,a) - F(b,b));
          }
        }
      }

      double sigma_hhp = 0.0;
      for (auto i = fc; i < ndocc; i++) {
        for (auto j = fc; j < ndocc; j++) {
          for (auto a = ndocc; a < nbasis; a++) {
            sigma_hhp += W(p,a,i,j)*(g(p,i,a,j))/(Ep_old + F(a,a) - F(i,i) - F(j,j))
                       + U(p,a,i,j)*(g(i,p,j,a))/(Ep_old + F(a,a) - F(i,i) - F(j,j));
          }
        }
      }

      Ep_new = F(p,p) + sigma_hpp + sigma_hhp;
      std::cout << Ep_new << std::endl;
      diff = Ep_new - Ep_old;
      Ep_old = Ep_new;
    }
    std::cout << "P3: "<< std::endl;
    std::cout << "F(3,3) = " << F(3,3) << " a.u.   " << F(3,3) * 27.2114 << " eV "<< std::endl;
    std::cout << "Ep_new = " << Ep_new << " a.u.   " << Ep_new * 27.2114 << " eV "<< std::endl;
*/


    bool NR2;
    NR2 = true;
    if(NR2) {

    }

  }
}

void gf2_test(const int nbasis, const int ndocc, Eigen::VectorXd &E_orb, TensorRank4 &g) {

  std::cout << "Starting with test GF procedure:" << std::endl;

  Eigen::MatrixXd F(nbasis,nbasis);
  F = Eigen::MatrixXd::Zero(nbasis,nbasis);
  for (auto i = 0; i < nbasis; i++) {
    for (auto j = 0; j < nbasis; j++) {
      if (i == j) {
        F(i,j) = E_orb(i);
      }
    }
  }




  bool SO = false;
  if (SO)
  {

    const int nso = 2*nbasis;
    const int ndocc_so = 2*ndocc;

    const int fc_so = 2;

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
    std::cout << "SO " << std::endl;
    std::cout << "NR2 self energy method:" << std::endl;

    int nvirt_so = nso-ndocc_so;

    std::cout << "number of all spin-orbitals: " << nso << std::endl;
    std::cout << "number of occupied spin-orbitals: " << ndocc_so << std::endl;
    std::cout << "number of unoccupied spin-orbitals: " << nvirt_so << std::endl;

    //definition of sizes
    int h = ndocc_so-fc_so;
    int p = nvirt_so;

    //H_h_p2h
    int p2h = p*h*(h-1)*0.5;
    std::cout << "p2h = " << p2h << std::endl;
    Eigen::MatrixXd H_h_p2h(h,p2h);
    H_h_p2h = Eigen::MatrixXd::Zero(h,p2h);

    int pp;
    for (auto i = fc_so; i < ndocc_so; i++) {
      pp = 0;
      for (auto a = ndocc_so; a < nso; a++) {
        for (auto k = fc_so; k < ndocc_so; k++) {
          for (auto j = fc_so; j < k; j++) {
            H_h_p2h(i-fc_so,pp) = g_so(i,j,a,k)-g_so(i,k,a,j);
            pp++;
          }
        }
      }
    }
    std::cout << "H_h_p2h = " << std::endl;

    Eigen::MatrixXd H_p2h_p2h(p2h,p2h);
    H_p2h_p2h = Eigen::MatrixXd::Zero(p2h,p2h);

    //H_p2h_h
    Eigen::MatrixXd H_p2h_h(p2h,h);
    H_p2h_h = Eigen::MatrixXd::Zero(p2h,h);

    pp = 0;
    for (auto a = ndocc_so; a < nso; a++) {
      for (auto k = fc_so; k < ndocc_so; k++) {
        for (auto j = fc_so; j < k; j++) {

          for (auto i = fc_so; i < ndocc_so; i++) {

            double sum = 0.0;
            for (auto b = ndocc_so; b < nso; b++) {
              for (auto c = ndocc_so; c < nso; c++) {
                sum -= 0.5*(g_so(b,i,c,a)-g_so(b,a,c,i))*(g_so(b,j,c,k)-g_so(b,k,c,j))/(Fso(b,b)+Fso(c,c)-Fso(j,j)-Fso(k,k));
              }
            }

            H_p2h_h(pp,i-fc_so) += sum;

            sum = 0.0;
            for (auto b = ndocc_so; b < nso; b++) {
              for (auto l = fc_so; l < ndocc_so; l++) {
                sum -= (g_so(b,i,j,l)-g_so(b,l,j,i))*(g_so(b,k,a,l)-g_so(b,l,a,k))/(Fso(b,b)+Fso(a,a)-Fso(k,k)-Fso(l,l))
                                                -(g_so(b,i,k,l)-g_so(b,l,k,i))*(g_so(b,j,a,l)-g_so(b,l,a,j))/(Fso(b,b)+Fso(a,a)-Fso(j,j)-Fso(l,l));
              }
            }
            H_p2h_h(pp,i-fc_so) += sum;

          }
          pp++;
        }
      }
    }
    std::cout << "H_p2h_h = " << std::endl;
    //std::cout << H_p2h_h << std::endl;

    //H_p_p2h
    Eigen::MatrixXd H_p_p2h(p,p2h);
    H_p_p2h = Eigen::MatrixXd::Zero(p,p2h);
    for (auto a = ndocc_so; a < nso; a++) {
      pp = 0;
      for (auto b = ndocc_so; b < nso; b++) {
        for (auto j = fc_so; j < ndocc_so; j++) {
          for (auto i = fc_so; i < j; i++) {

            H_p_p2h(a-ndocc_so,pp) = g_so(a,i,b,j) - g_so(a,j,b,i);
            pp++;
          }
        }
      }
    }
    std::cout << "H_p_p2h = " << std::endl;
    //std::cout << H_p_p2h << std::endl;

    Eigen::MatrixXd H_p2h_p(p2h,p);
    H_p2h_p = H_p_p2h.transpose();
    std::cout << "H_p2h_p = " << std::endl;
    //std::cout << H_p2h_p << std::endl;

    int orbital = 6-fc_so;
    double E_old = Fso(orbital,orbital);
    double E_new;
    double diff = 1.0;

    while (fabs(diff)>1e-05) {
      pp = 0;
      for (auto a = ndocc_so; a < nso; a++) {
        for (auto i = fc_so; i < ndocc_so; i++) {
          for (auto j = fc_so; j < i; j++) {

            int qq = 0;
            for (auto b = ndocc_so; b < nso; b++) {
              for (auto k = fc_so; k < ndocc_so; k++) {
                for (auto l = fc_so; l < k; l++) {

                  H_p2h_p2h(pp,qq) = E_old*(a == b)*(i == k)*(j == l) - ((a == b)*(i == k)*(j == l)*(Fso(i,i)+Fso(j,j)-Fso(a,a))
                      -(a == b)*(g_so(i,k,j,l)-g_so(i,l,j,k)) + (l == j)*(g_so(b,a,i,k)-g_so(b,k,i,a))
                      -(l == i)*(g_so(b,a,j,k)-g_so(b,k,j,a))
                      - (j == k)*(g_so(b,a,i,l)-g_so(b,l,i,a))
                      +(k == i)*(g_so(b,a,j,l)-g_so(b,l,j,a)));

                  qq++;
                }
              }
            }
            pp++;
          }
        }
      }
      std::cout << "H_p2h_p2h = " << std::endl;
      //std::cout << H_p2h_p2h << std::endl;

      Eigen::MatrixXd x1 = H_p2h_p2h.ldlt().solve(H_h_p2h.transpose()+0.5*H_p2h_h);

      Eigen::MatrixXd Sigma_ij(h,h);
      Sigma_ij = Eigen::MatrixXd::Zero(h,h);
      Sigma_ij = (H_h_p2h+0.5*H_p2h_h.transpose())*x1;
      //Sigma_ij = (H_h_p2h)*x1;
      //Sigma_ij = (0.5*H_p2h_h.transpose())*x1;

      std::cout << "Sigma_ij = " << std::endl;
      //std::cout << Sigma_ij << std::endl;

      Eigen::MatrixXd Sigma_aj(p,h);
      Sigma_aj = Eigen::MatrixXd::Zero(p,h);

      Sigma_aj = H_p_p2h*x1;

      std::cout << "Sigma_aj = " << std::endl;
      //std::cout << Sigma_aj << std::endl;

      Eigen::MatrixXd Sigma_ia(h,p);
      Sigma_ia = Eigen::MatrixXd::Zero(h,p);
      Sigma_ia = Sigma_aj.transpose();

      std::cout << "Sigma_ia = " << std::endl;
      //std::cout << Sigma_ia << std::endl;


      Eigen::MatrixXd x2 = H_p2h_p2h.ldlt().solve(H_p2h_p);

      Eigen::MatrixXd Sigma_ab(p,p);
      Sigma_ab = Eigen::MatrixXd::Zero(p,p);
      Sigma_ab = H_p_p2h*x2;

      std::cout << "Sigma_ab = " << std::endl;
      //std::cout << Sigma_ab << std::endl;

      Eigen::MatrixXd Sigma(h+p,h+p);
      Sigma = Eigen::MatrixXd::Zero(h+p,h+p);


      for (int pp = 0; pp < h+p; pp++) {
        for (int qq = 0; qq < h+p; qq++) {

          //H_h_h block
          if (pp < h && qq < h) {
            Sigma(pp,qq) = Sigma_ij(pp,qq);
          }

          //H_h_p block
          if (pp < h && qq >= h) {
            Sigma(pp,qq) = Sigma_ia(pp,qq-h);
          }

          //H_p_h block
          if (pp >= h && qq < h) {
            Sigma(pp,qq) = Sigma_aj(pp-h,qq);
          }

          //H_p_p block
          if (pp >= h && qq >= h) {
            Sigma(pp,qq) = Sigma_ab(pp-h,qq-h);
          }

        }
      }

      std::cout << "Sigma = " << std::endl;
      //std::cout << Sigma << std::endl;

      for (auto pp = fc_so; pp < nso; pp++) {
        for (auto qq = fc_so; qq < nso; qq++) {

          double sigma_kk_hpp = 0;
          for (auto i = fc_so; i < ndocc_so; i++) {
            for (auto a = ndocc_so; a < nso; a++) {
              for (auto b = ndocc_so; b < a; b++) {
                sigma_kk_hpp += (g_so(pp,a,i,b) - g_so(pp,b,i,a)) * (g_so(a,qq,b,i) - g_so(a,i,b,qq))/
                    (E_old + Fso(i,i) - Fso(a,a) - Fso(b,b));
              }
            }
          }

          Sigma(pp-fc_so,qq-fc_so) += sigma_kk_hpp + Fso(pp,qq);

        }
      }

      //std::cout << Sigma << std::endl;

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma);
      Eigen::VectorXd Ep = es.eigenvalues();

      E_new = Ep(orbital);

      diff = E_new - E_old;
      std::cout << "diff = " << diff << "  E_new = " << E_new << std::endl;
      E_old = E_new;
    }

    std::cout << "NR2: "<< std::endl;
    std::cout << "E_new = " << E_new << " a.u.   " << E_new * 27.21138602 << " eV "<< std::endl;
    printf("Ek_new %20.12f \n", E_new*27.21138602);

  }
  {
    std::cout << "NR2 self energy method in spin-free form:" << std::endl;

    int nvirt = nbasis-ndocc;

    std::cout << "number of all orbitals: " << nbasis << std::endl;
    std::cout << "number of occupied orbitals: " << ndocc << std::endl;
    std::cout << "number of unoccupied orbitals: " << nvirt << std::endl;

    TensorRank4 t2(nbasis,nbasis,nbasis,nbasis);
    for (auto i = 0; i < ndocc; i++) {
      for (auto j = 0; j < ndocc; j++) {
        for (auto a = ndocc; a < nbasis; a++) {
          for (auto b = ndocc; b < nbasis; b++) {
            t2(i,a,j,b)= g(i,a,j,b)/(E_orb(i)+E_orb(j)-E_orb(a)-E_orb(b));
          }
        }
      }
    }

    int fc = 0;
    //definition of sizes
    const int h = ndocc-fc;
    const int p = nvirt;

    //H_h_p2h
    int p2h = p*h*h;
    std::cout << "p2h = " << p2h << std::endl;
    Eigen::MatrixXd H_h_p2h(h,p2h);
    H_h_p2h = Eigen::MatrixXd::Zero(h,p2h);

    int pp;
    for (auto i = fc; i < ndocc; i++) {
      pp = 0;
      for (auto a = ndocc; a < nbasis; a++) {
        for (auto k = fc; k < ndocc; k++) {
          for (auto j = fc; j < ndocc; j++) {
            H_h_p2h(i-fc,pp) = 4.0*g(i,j,a,k)-2.0*g(i,k,a,j);
            pp++;
          }
        }
      }
    }
    std::cout << "H_h_p2h = " << std::endl;
    //std::cout << H_h_p2h << std::endl;

    //H_p2h_h
    Eigen::MatrixXd H_p2h_h(p2h,h);
    H_p2h_h = Eigen::MatrixXd::Zero(p2h,h);

    pp = 0;
    for (auto a = ndocc; a < nbasis; a++) {
      for (auto j = fc; j < ndocc; j++) {
        for (auto k = fc; k < ndocc; k++) {

          for (auto i = fc; i < ndocc; i++) {

            double sum = 0.0;
            for (auto b = ndocc; b < nbasis; b++) {
              for (auto c = ndocc; c < nbasis; c++) {
                sum += 2.0*g(b,a,c,i)*t2(j,b,k,c)+2.0*g(c,a,b,i)*t2(k,b,j,c)-g(c,a,b,i)*t2(j,b,k,c)-g(b,a,c,i)*t2(k,b,j,c);
              }
            }

            H_p2h_h(pp,i-fc) += sum;

            sum = 0.0;
            for (auto b = ndocc; b < nbasis; b++) {
              for (auto l = fc; l < ndocc; l++) {
                sum += g(b,i,k,l)*t2(j,b,l,a)+4.0*g(b,l,i,k)*t2(l,b,j,a)-2.0*g(b,l,k,i)*t2(j,b,l,a)-2.0*g(b,i,k,l)*t2(l,b,j,a)
                      +g(b,l,j,i)*t2(k,b,l,a)+g(b,i,j,l)*t2(l,b,k,a)-2.0*g(b,i,j,l)*t2(k,b,l,a)-2.0*g(b,l,j,i)*t2(l,b,k,a);
              }
            }
            H_p2h_h(pp,i-fc) += 2.0*sum;

          }
          pp++;
        }
      }
    }

    std::cout << "H_p2h_h = " << std::endl;
    //std::cout << H_p2h_h << std::endl;

    //H_p_p2h
    Eigen::MatrixXd H_p_p2h(p,p2h);
    H_p_p2h = Eigen::MatrixXd::Zero(p,p2h);
    for (auto a = ndocc; a < nbasis; a++) {
      pp = 0;
      for (auto b = ndocc; b < nbasis; b++) {
        for (auto j = fc; j < ndocc; j++) {
          for (auto i = fc; i < ndocc; i++) {
            H_p_p2h(a-ndocc,pp) = 4.0*g(a,i,b,j) - 2.0*g(a,j,b,i);
            pp++;
          }
        }
      }
    }
    std::cout << "H_p_p2h = " << std::endl;
    //std::cout << H_p_p2h << std::endl;

    Eigen::MatrixXd H_p2h_p(p2h,p);
    H_p2h_p = H_p_p2h.transpose();
    std::cout << "H_p2h_p = " << std::endl;
    //std::cout << H_p2h_p << std::endl;


    int orbital = 4-fc;
    double E_old = F(orbital,orbital);
    double E_new;
    double diff = 1.0;

    while (fabs(diff)>1e-05) {
      //H_p2h_p2h
      Eigen::MatrixXd H_p2h_p2h(p2h,p2h);
      H_p2h_p2h = Eigen::MatrixXd::Zero(p2h,p2h);

      pp = 0;
      for (auto a = ndocc; a < nbasis; a++) {
        for (auto i = fc; i < ndocc; i++) {
          for (auto j = fc; j < ndocc; j++) {

            int qq = 0;
            for (auto b = ndocc; b < nbasis; b++) {
              for (auto k = fc; k < ndocc; k++) {
                for (auto l = fc; l < ndocc; l++) {

                  H_p2h_p2h(pp,qq) = E_old*(4.0*(a == b)*(i == k)*(j == l) - 2.0*(a == b)*(i == l)*(j == k)) -
                      (4.0*(a == b)*(i == k)*(j == l) - 2.0*(a == b)*(i == l)*(j == k))*(F(i,i)+F(j,j)-F(a,a))
                      -(-(a == b)*(2.0*g(i,k,j,l)-1.0*g(i,l,j,k)) + (l == j)*(2.0*g(a,b,k,i)-4.0*g(k,b,a,i))
                          +(l == i)*(2.0*g(b,k,j,a)-1.0*g(b,a,j,k))
                          +(j == k)*(2.0*g(b,l,i,a)-1.0*g(b,a,i,l))
                          +(k == i)*(2.0*g(b,a,j,l)-1.0*g(b,l,j,a)))*2.0;

                  qq++;
                }
              }
            }
            pp++;
          }
        }
      }
      std::cout << "H_p2h_p2h = " << std::endl;
      //std::cout << H_p2h_p2h << std::endl;

      Eigen::MatrixXd x1 = H_p2h_p2h.ldlt().solve(H_h_p2h.transpose()+0.5*H_p2h_h);

      Eigen::MatrixXd Sigma_ij(h,h);
      Sigma_ij = Eigen::MatrixXd::Zero(h,h);
      Sigma_ij = 0.5*(H_h_p2h+0.5*H_p2h_h.transpose())*x1;

      std::cout << "Sigma_ij = " << std::endl;
      //std::cout << Sigma_ij << std::endl;

      Eigen::MatrixXd Sigma_aj(p,h);
      Sigma_aj = Eigen::MatrixXd::Zero(p,h);

      Sigma_aj = 0.5*H_p_p2h*x1;

      std::cout << "Sigma_aj = " << std::endl;
      //std::cout << Sigma_aj << std::endl;

      Eigen::MatrixXd Sigma_ia(h,p);
      Sigma_ia = Eigen::MatrixXd::Zero(h,p);
      Sigma_ia = Sigma_aj.transpose();

      std::cout << "Sigma_ia = " << std::endl;
      //std::cout << Sigma_ia << std::endl;

      Eigen::MatrixXd x2 = H_p2h_p2h.ldlt().solve(H_p2h_p);

      Eigen::MatrixXd Sigma_ab(p,p);
      Sigma_ab = Eigen::MatrixXd::Zero(p,p);
      Sigma_ab = 0.5*H_p_p2h*x2;

      std::cout << "Sigma_ab = " << std::endl;
      //std::cout << Sigma_ab << std::endl;

      Eigen::MatrixXd Sigma(h+p,h+p);
      Sigma = Eigen::MatrixXd::Zero(h+p,h+p);


      for (int pp = 0; pp < h+p; pp++) {
        for (int qq = 0; qq < h+p; qq++) {

          //H_h_h block
          if (pp < h && qq < h) {
            Sigma(pp,qq) = Sigma_ij(pp,qq);
          }

          //H_h_p block
          if (pp < h && qq >= h) {
            Sigma(pp,qq) = Sigma_ia(pp,qq-h);
          }

          //H_p_h block
          if (pp >= h && qq < h) {
            Sigma(pp,qq) = Sigma_aj(pp-h,qq);
          }

          //H_p_p block
          if (pp >= h && qq >= h) {
            Sigma(pp,qq) = Sigma_ab(pp-h,qq-h);
          }

        }
      }

      std::cout << "Sigma = " << std::endl;
      //std::cout << Sigma << std::endl;

      for (auto pp = fc; pp < nbasis; pp++) {
        for (auto qq = fc; qq < nbasis; qq++) {

          double sigma_kk_hpp = 0;
          for (auto i = fc; i < ndocc; i++) {
            for (auto a = ndocc; a < nbasis; a++) {
              for (auto b = ndocc; b < nbasis; b++) {
                sigma_kk_hpp += (g(pp,a,i,b)) * (2.0*g(a,qq,b,i) - g(a,i,b,qq))/
                    (E_old + F(i,i) - F(a,a) - F(b,b));
              }
            }
          }

          Sigma(pp-fc,qq-fc) += sigma_kk_hpp + F(pp,qq);

        }
      }

      //std::cout << Sigma << std::endl;

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma);
      Eigen::VectorXd Ep = es.eigenvalues();

      E_new = Ep(orbital);

      diff = E_new - E_old;
      std::cout << "diff = " << diff << "  E_new = " << E_new << std::endl;
      E_old = E_new;
    }

    std::cout << "NR2: "<< std::endl;
    std::cout << "E_new = " << E_new << " a.u.   " << E_new * 27.21138602 << " eV "<< std::endl;
    printf("Ek_new %20.12f \n", E_new*27.21138602);
  }

  {
    std::cout << "GF2 self energy method in spin-free form by lin-solve:" << std::endl;

    int nvirt = nbasis-ndocc;

    std::cout << "number of all orbitals: " << nbasis << std::endl;
    std::cout << "number of occupied orbitals: " << ndocc << std::endl;
    std::cout << "number of unoccupied orbitals: " << nvirt << std::endl;

    int fc = 0;
    //definition of sizes
    const int h = ndocc-fc;
    const int p = nvirt;

    //H_h_p2h
    int p2h = p*h*h;
    std::cout << "p2h = " << p2h << std::endl;
    Eigen::MatrixXd H_h_p2h(h+p,p2h);
    H_h_p2h = Eigen::MatrixXd::Zero(h+p,p2h);

    int pp;
    for (auto qq = fc; qq < nbasis; qq++) {
      pp = 0;
      for (auto a = ndocc; a < nbasis; a++) {
        for (auto k = fc; k < ndocc; k++) {
          for (auto j = fc; j < ndocc; j++) {
            H_h_p2h(qq-fc,pp) = -4.0*g(qq,k,a,j)+2.0*g(qq,j,a,k);
            pp++;
          }
        }
      }
    }
    std::cout << "H_h_p2h = " << std::endl;
    //std::cout << H_h_p2h << std::endl;

    int orbital = 4-fc;
    double E_old = F(orbital,orbital);
    double E_new;
    double diff = 1.0;

    while (fabs(diff)>1e-05) {
      //H_p2h_p2h
      Eigen::MatrixXd H_p2h_p2h(p2h,p2h);
      H_p2h_p2h = Eigen::MatrixXd::Zero(p2h,p2h);

      pp = 0;
      for (auto a = ndocc; a < nbasis; a++) {
        for (auto i = fc; i < ndocc; i++) {
          for (auto j = fc; j < ndocc; j++) {

            int qq = 0;
            for (auto b = ndocc; b < nbasis; b++) {
              for (auto k = fc; k < ndocc; k++) {
                for (auto l = fc; l < ndocc; l++) {

                  H_p2h_p2h(pp,qq) = E_old*(8.0*(a == b)*(i == k)*(j == l) - 4.0*(a == b)*(i == l)*(j == k)) -
                      (8.0*(a == b)*(i == k)*(j == l) - 4.0*(a == b)*(i == l)*(j == k))*(F(i,i)+F(j,j)-F(a,a));

                  qq++;
                }
              }
            }
            pp++;
          }
        }
      }
      std::cout << "H_p2h_p2h = " << std::endl;
      //std::cout << H_p2h_p2h << std::endl;

      Eigen::MatrixXd x1 = H_p2h_p2h.ldlt().solve(H_h_p2h.transpose());

      Eigen::MatrixXd Sigma(h+p,h+p);
      Sigma = Eigen::MatrixXd::Zero(h+p,h+p);

      Sigma = (H_h_p2h)*x1;

      //std::cout << "Sigma = " << std::endl;
      //std::cout << Sigma << std::endl;

      for (auto pp = fc; pp < nbasis; pp++) {
        for (auto qq = fc; qq < nbasis; qq++) {

          double sigma_kk_hpp = 0;
          for (auto i = fc; i < ndocc; i++) {
            for (auto a = ndocc; a < nbasis; a++) {
              for (auto b = ndocc; b < nbasis; b++) {
                sigma_kk_hpp += (g(pp,a,i,b)) * (2.0*g(a,qq,b,i) - g(a,i,b,qq))/
                    (E_old + F(i,i) - F(a,a) - F(b,b));
              }
            }
          }

          //Sigma(pp-fc,qq-fc) += sigma_kk_hpp + F(pp,qq);
          Sigma(pp-fc,qq-fc) += F(pp,qq);

        }
      }

      //std::cout << Sigma << std::endl;

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma);
      Eigen::VectorXd Ep = es.eigenvalues();

      E_new = Ep(orbital);

      diff = E_new - E_old;
      std::cout << "diff = " << diff << "  E_new = " << E_new << std::endl;
      E_old = E_new;
    }

    std::cout << "GF2: "<< std::endl;
    std::cout << "E_new = " << E_new << " a.u.   " << E_new * 27.21138602 << " eV "<< std::endl;
    printf("Ek_new %20.12f \n", E_new*27.21138602);
  }

}
