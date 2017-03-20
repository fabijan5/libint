#include "tensor.h"

TensorRank4 ao_to_mo_integral_transform(const int nbasis,
                                        const int ndocc,
                                        const Eigen::MatrixXd &C,
                                        const TensorRank4 &g);

double mp2_energy(const int nbasis,
                  const int ndocc,
                  Eigen::VectorXd &E_orb,
                  TensorRank4 &g);

void lt_mp2_energy(const int nbasis,
                   const int ndocc,
                   Eigen::VectorXd E_orb,
                   TensorRank4 &g,
                   double E_MP2_can);

void lt_ao_mp2_energy(const int nbasis,
                      const int ndocc,
                      Eigen::VectorXd E_orb,
                      Eigen::MatrixXd &C,
                      TensorRank4 &g,
                      double E_MP2_can);

void gf2(const int nbasis,
                  const int ndocc,
                  Eigen::VectorXd &E_orb,
                  TensorRank4 &g);

void gf2_test(const int nbasis,
              const int ndocc,
              Eigen::VectorXd &E_orb,
              TensorRank4 &g);
