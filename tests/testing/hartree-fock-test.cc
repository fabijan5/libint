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

// Libint Gaussian integrals library
#include <libint2.hpp>

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

struct Atom {
    int atomic_number;
    double x, y, z;
};

std::vector<Atom> read_geometry(const std::string& filename);
std::vector<libint2::Shell> make_sto3g_basis(const std::vector<Atom>& atoms);
std::vector<libint2::Shell> make_cc_pvdz_basis(const std::vector<Atom>& atoms);
size_t nbasis(const std::vector<libint2::Shell>& shells);
std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell>& shells);
Matrix compute_1body_ints(const std::vector<libint2::Shell>& shells,
                          libint2::Operator t,
                          const std::vector<Atom>& atoms = std::vector<Atom>());
TensorRank4 compute_2body_ints(const std::vector<libint2::Shell>& shells);


Eigen::MatrixXd symmetric_orthogonalization(const Eigen::MatrixXd &S) {

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
  Eigen::VectorXd D = es.eigenvalues();
  Eigen::MatrixXd V = es.eigenvectors();

  for (auto i = 0; i < D.rows(); i++)
    D(i) = pow(D(i),-0.5);

  Eigen::MatrixXd X;
  X = V*D.asDiagonal()*V.transpose();

  return X;
}

Eigen::MatrixXd fock_build(const int nbasis, Eigen::MatrixXd &P, const Eigen::MatrixXd &H_core, const TensorRank4 &g) {

  Eigen::MatrixXd F(nbasis, nbasis);

  for (auto mu = 0; mu < nbasis; mu++) {
    for (auto nu = 0; nu < nbasis; nu++) {
      double integral = 0.0;
      for (auto rho = 0; rho < nbasis; rho++) {
        for (auto sigma = 0; sigma < nbasis; sigma++) {
          integral += P(rho,sigma)*(2.0*g(mu,nu,rho,sigma)-1.0*g(mu,rho,nu,sigma));
        }
      }
      F(mu,nu) = H_core(mu,nu) + integral;
    }
  }

  return F;
}

Eigen::MatrixXd occupied_slice_of_MO_coeff(const int nbasis, const int ndocc, Eigen::MatrixXd &C) {

  Eigen::MatrixXd C_occ(nbasis,ndocc);

  for (auto mu = 0; mu < nbasis; mu++) {
    for (auto i = 0; i < ndocc; i++) {
      C_occ(mu,i) = C(mu,i);
    }
  }
  return C_occ;
}

double norm_P(Eigen::MatrixXd &P, Eigen::MatrixXd &P_guess) {
  double norm = 0.0;

  for (auto mu = 0; mu < P.rows(); mu++) {
    for (auto nu = 0; nu < P.cols(); nu++) {
      norm += sqrt(pow(P(mu,nu) - P_guess(mu,nu),2));
    }
  }

  return norm;
}

double compute_energy(const int nbasis, Eigen::MatrixXd &P, const Eigen::MatrixXd &H_core, Eigen::MatrixXd &F) {

  double E_electronic = 0.0;

  for (auto mu = 0; mu < nbasis; mu++) {
    for (auto nu = 0; nu < nbasis; nu++) {
      E_electronic += P(mu,nu)*(H_core(mu,nu) + F(mu,nu));
    }
  }

  return E_electronic;
}

Eigen::MatrixXd hartree_fock(const double enuc, const int nbasis, const int ndocc, const Eigen::MatrixXd &S,
                             const Eigen::MatrixXd &H_core, const TensorRank4 &g, Eigen::VectorXd &E_orb) {

  //perform symmetric orthogonalization
  Eigen::MatrixXd X = symmetric_orthogonalization(S);
  Eigen::MatrixXd C;
  Eigen::MatrixXd P_guess;
  P_guess.setZero(nbasis,nbasis);

  double norm = 1.0;
  double threshold = 1e-08;
  double E_electronic;
  int count = 0;

  printf(" Hartree-Fock Energy\n");
  while (norm > threshold) {

    count++;
    Eigen::MatrixXd F(nbasis, nbasis);
    F = fock_build(nbasis, P_guess, H_core, g);

    Eigen::MatrixXd Ft = X.transpose()*F*X;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Ft);
    Eigen::MatrixXd Ct = es.eigenvectors();
    E_orb = es.eigenvalues();

    C = X*Ct;
    Eigen::MatrixXd C_occ = occupied_slice_of_MO_coeff(nbasis,ndocc,C);

    Eigen::MatrixXd P = C_occ*C_occ.transpose();
    norm = norm_P(P,P_guess);
    P_guess = 0.2*P_guess+0.8*P;
    E_electronic = compute_energy(nbasis, P, H_core, F);

    if (count == 1) {
    printf(" Iter         E_total            norm\n");}
    printf("  %02d %20.12f %20.12e\n", count, E_electronic + enuc, norm);
  }

  return C;
}

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

int main(int argc, char *argv[]) {

  using std::cout;
  using std::cerr;
  using std::endl;

  using libint2::Shell;
  using libint2::Engine;
  using libint2::Operator;

  /*** =========================== ***/
  /*** initialize molecule         ***/
  /*** =========================== ***/

  // read geometry from a file; by default read from h2o.xyz, else take filename (.xyz) from the command line
  const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";
  std::vector<Atom> atoms = read_geometry(filename);

  // count the number of electrons
  auto nelectron = 0;
  for (auto i = 0; i < atoms.size(); ++i)
    nelectron += atoms[i].atomic_number;
  const auto ndocc = nelectron / 2;

  // compute the nuclear repulsion energy
  auto enuc = 0.0;
  for (auto i = 0; i < atoms.size(); i++)
    for (auto j = i + 1; j < atoms.size(); j++) {
      auto xij = atoms[i].x - atoms[j].x;
      auto yij = atoms[i].y - atoms[j].y;
      auto zij = atoms[i].z - atoms[j].z;
      auto r2 = xij*xij + yij*yij + zij*zij;
      auto r = sqrt(r2);
      enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
    }

  /*** =========================== ***/
  /*** create basis set            ***/
  /*** =========================== ***/

  //auto shells = make_sto3g_basis(atoms);
  auto shells = make_cc_pvdz_basis(atoms);

  size_t nao = 0;
  for (auto s=0; s<shells.size(); ++s)
    nao += shells[s].size();

  /*** =========================== ***/
  /*** compute 1-e integrals       ***/
  /*** =========================== ***/

  // initializes the Libint integrals library ... now ready to compute
  libint2::initialize();

  // compute overlap integrals
  auto S = compute_1body_ints(shells, Operator::overlap);

  // compute kinetic-energy integrals
  auto T = compute_1body_ints(shells, Operator::kinetic);

  // compute nuclear-attraction integrals
  Matrix V = compute_1body_ints(shells, Operator::nuclear, atoms);

  // Core Hamiltonian = T + V
  Matrix H_core = T + V;

  // T and V no longer needed, free up the memory
  T.resize(0,0);
  V.resize(0,0);

  TensorRank4 g = compute_2body_ints(shells);
  Eigen::VectorXd E_orb;

  Eigen::MatrixXd C = hartree_fock(enuc, nao, ndocc, S, H_core, g, E_orb);
  TensorRank4 g_mo = ao_to_mo_integral_transform(nao, ndocc, C, g);

  double E_MP2 = mp2_energy(nao, ndocc, E_orb, g_mo);
  printf("\n");
  printf("E_MP2 = %20.12f\n", E_MP2);

  lt_mp2_energy(nao, ndocc, E_orb, g_mo, E_MP2);

return 0;
}

// this reads the geometry in the standard xyz format supported by most chemistry software
std::vector<Atom> read_dotxyz(std::istream& is) {
  // line 1 = # of atoms
  size_t natom;
  is >> natom;
  // read off the rest of line 1 and discard
  std::string rest_of_line;
  std::getline(is, rest_of_line);

  // line 2 = comment (possibly empty)
  std::string comment;
  std::getline(is, comment);

  std::vector<Atom> atoms(natom);
  for (auto i = 0; i < natom; i++) {
    std::string element_label;
    double x, y, z;
    is >> element_label >> x >> y >> z;

    // .xyz files report element labels, hence convert to atomic numbers
    int Z;
    if (element_label == "H")
      Z = 1;
    else if (element_label == "C")
      Z = 6;
    else if (element_label == "N")
      Z = 7;
    else if (element_label == "O")
      Z = 8;
    else if (element_label == "F")
      Z = 9;
    else if (element_label == "S")
      Z = 16;
    else if (element_label == "Cl")
      Z = 17;
    else {
      std::cerr << "read_dotxyz: element label \"" << element_label << "\" is not recognized" << std::endl;
      throw "Did not recognize element label in .xyz file";
    }

    atoms[i].atomic_number = Z;

    // .xyz files report Cartesian coordinates in angstroms; convert to bohr
    const auto angstrom_to_bohr = 1 / 0.52917721092; // 2010 CODATA value
    atoms[i].x = x * angstrom_to_bohr;
    atoms[i].y = y * angstrom_to_bohr;
    atoms[i].z = z * angstrom_to_bohr;
  }

  return atoms;
}

std::vector<Atom> read_geometry(const std::string& filename) {

  std::cout << "Will read geometry from " << filename << std::endl;
  std::ifstream is(filename);
  assert(is.good());

  // to prepare for MPI parallelization, we will read the entire file into a string that can be
  // broadcast to everyone, then converted to an std::istringstream object that can be used just like std::ifstream
  std::ostringstream oss;
  oss << is.rdbuf();
  // use ss.str() to get the entire contents of the file as an std::string
  // broadcast
  // then make an std::istringstream in each process
  std::istringstream iss(oss.str());

  // check the extension: if .xyz, assume the standard XYZ format, otherwise throw an exception
  if ( filename.rfind(".xyz") != std::string::npos)
    return read_dotxyz(iss);
  else
    throw "only .xyz files are accepted";
}

std::vector<libint2::Shell> make_sto3g_basis(const std::vector<Atom>& atoms) {

  using libint2::Shell;

  std::vector<Shell> shells;

  for(auto a=0; a<atoms.size(); ++a) {

    // STO-3G basis set
    // cite: W. J. Hehre, R. F. Stewart, and J. A. Pople, The Journal of Chemical Physics 51, 2657 (1969)
    //       doi: 10.1063/1.1672392
    // obtained from https://bse.pnl.gov/bse/portal
    switch (atoms[a].atomic_number) {
      case 1: // Z=1: hydrogen
        shells.push_back(
            {
              {3.425250910, 0.623913730, 0.168855400}, // exponents of primitive Gaussians
              {  // contraction 0: s shell (l=0), spherical=false, contraction coefficients
                {0, false, {0.15432897, 0.53532814, 0.44463454}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}   // origin coordinates
            }
        );
        break;

      case 6: // Z=6: carbon
        shells.push_back(
            {
              {71.616837000, 13.045096000, 3.530512200},
              {
                {0, false, {0.15432897, 0.53532814, 0.44463454}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {2.941249400, 0.683483100, 0.222289900},
              {
                {0, false, {-0.09996723, 0.39951283, 0.70011547}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {2.941249400, 0.683483100, 0.222289900},
              { // contraction 0: p shell (l=1), spherical=false
                {1, false, {0.15591627, 0.60768372, 0.39195739}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        break;

      case 7: // Z=7: nitrogen
        shells.push_back(
            {
              {99.106169000, 18.052312000, 4.885660200},
              {
                {0, false, {0.15432897, 0.53532814, 0.44463454}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {3.780455900, 0.878496600, 0.285714400},
              {
                {0, false, {-0.09996723, 0.39951283, 0.70011547}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
          {3.780455900, 0.878496600, 0.285714400},
              { // contraction 0: p shell (l=1), spherical=false
                {1, false, {0.15591627, 0.60768372, 0.39195739}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        break;

      case 8: // Z=8: oxygen
        shells.push_back(
            {
              {130.709320000, 23.808861000, 6.443608300},
              {
                {0, false, {0.15432897, 0.53532814, 0.44463454}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {5.033151300, 1.169596100, 0.380389000},
              {
                {0, false, {-0.09996723, 0.39951283, 0.70011547}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {5.033151300, 1.169596100, 0.380389000},
              { // contraction 0: p shell (l=1), spherical=false
                {1, false, {0.15591627, 0.60768372, 0.39195739}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        break;

      default:
        throw "do not know STO-3G basis for this Z";
    }

  }

  return shells;
}

std::vector<libint2::Shell> make_cc_pvdz_basis(const std::vector<Atom>& atoms) {
    using libint2::Shell;

    std::vector<Shell> shells;

    for(auto a=0; a<atoms.size(); ++a) {

      // cc-pVDZ basis set
      // obtained from https://bse.pnl.gov/bse/portal
      switch (atoms[a].atomic_number) {
        case 1: // Z=1: hydrogen
          shells.push_back(
              {
                {13.0100000, 1.9620000, 0.4446000, 0.1220000}, // exponents of primitive Gaussians
                {  // contraction 0: s shell (l=0), spherical=false, contraction coefficients
                  {0, false, {0.0196850, 0.1379770, 0.4781480, 0.5012400}}
                },
                {{atoms[a].x, atoms[a].y, atoms[a].z}}   // origin coordinates
              }
          );
          shells.push_back(
              {
                {0.1220000},
                {
                  {0, false, {1.0000000}}
                },
                {{atoms[a].x, atoms[a].y, atoms[a].z}}
              }
          );
          shells.push_back(
              {
                {0.7270000},
                {
                  {1, false, {1.0000000}}
                },
                {{atoms[a].x, atoms[a].y, atoms[a].z}}
              }
          );

          break;

      case 6: // Z=6: carbon
        shells.push_back(
            {
              {6665.0000000, 1000.0000000, 228.0000000, 64.7100000, 21.0600000, 7.4950000, 2.7970000, 0.5215000, 0.1596000}, // exponents of primitive Gaussians
              {  // contraction 0: s shell (l=0), spherical=false, contraction coefficients
                {0, false, {0.0006920, 0.0053290, 0.0270770, 0.1017180, 0.2747400, 0.4485640, 0.2850740, 0.0152040, -0.0031910}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}   // origin coordinates
            }
        );
        shells.push_back(
            {
              {6665.0000000, 1000.0000000, 228.0000000, 64.7100000, 21.0600000, 7.4950000, 2.7970000, 0.5215000, 0.1596000}, // exponents of primitive Gaussians
              {
                {0, false, {-0.0001460, -0.0011540, -0.0057250, -0.0233120, -0.0639550, -0.1499810, -0.1272620, 0.5445290, 0.5804960}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {0.1596000},
              {
                {0, false, {1.0000000}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {9.4390000, 2.0020000, 0.5456000, 0.1517000},
              {
                {1, false, {0.0381090, 0.2094800, 0.5085570, 0.4688420}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {0.1517000},
              {
                {1, false, {1.0000000}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {0.5500000},
              {
                {2, true, {1.0000000}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );

          break;

      case 8: // Z=8: oxygen
        shells.push_back(
            {
              {11720.0000000, 1759.0000000, 400.8000000, 113.7000000, 37.0300000, 13.2700000, 5.0250000, 1.0130000, 0.3023000}, // exponents of primitive Gaussians
              {  // contraction 0: s shell (l=0), spherical=false, contraction coefficients
                {0, false, {0.0007100, 0.0054700, 0.0278370, 0.1048000, 0.2830620, 0.4487190, 0.2709520, 0.0154580, -0.0025850}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}   // origin coordinates
            }
        );
        shells.push_back(
            {
              {11720.0000000, 1759.0000000, 400.8000000, 113.7000000, 37.0300000, 13.2700000, 5.0250000, 1.0130000, 0.3023000}, // exponents of primitive Gaussians
              {
                {0, false, {-0.0001600, -0.0012630, -0.0062670, -0.0257160, -0.0709240, -0.1654110, -0.1169550, 0.5573680, 0.5727590}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {0.3023000},
              {
                {0, false, {1.0000000}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {17.7000000, 3.8540000, 1.0460000, 0.2753000},
              {
                {1, false, {0.0430180, 0.2289130, 0.5087280, 0.4605310}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {0.2753000},
              {
                {1, false, {1.0000000}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {1.1850000},
              {
                {2, true, {1.0000000}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );

          break;

      case 9: // Z=9: fluorine
        shells.push_back(
            {
              {14710.0000000, 2207.0000000, 502.8000000, 142.6000000, 46.4700000, 16.7000000, 6.3560000, 1.3160000, 0.3897000}, // exponents of primitive Gaussians
              {  // contraction 0: s shell (l=0), spherical=false, contraction coefficients
                {0, false, {0.0007210, 0.0055530, 0.0282670, 0.1064440, 0.2868140, 0.4486410, 0.2647610, 0.0153330, -0.0023320}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}   // origin coordinates
            }
        );
        shells.push_back(
            {
              {14710.0000000, 2207.0000000, 502.8000000, 142.6000000, 46.4700000, 16.7000000, 6.3560000, 1.3160000, 0.3897000},
              {
                {0, false, {-0.0001650, -0.0013080, -0.0064950, -0.0266910, -0.0736900, -0.1707760, -0.1123270, 0.5628140, 0.5687780}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {0.3897000},
              {
                {0, false, {1.0000000}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {22.6700000, 4.9770000, 1.3470000, 0.3471000},
              {
                {1, false, {0.0448780, 0.2357180, 0.5085210, 0.4581200}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {0.3471000},
              {
                {1, false, {1.0000000}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );
        shells.push_back(
            {
              {1.6400000},
              {
                {2, true, {1.0000000}}
              },
              {{atoms[a].x, atoms[a].y, atoms[a].z}}
            }
        );

          break;

        default:
          throw "do not know cc-pVDZ basis for this Z";
      }
    }

    return shells;

}

size_t nbasis(const std::vector<libint2::Shell>& shells) {
  size_t n = 0;
  for (const auto& shell: shells)
    n += shell.size();
  return n;
}

size_t max_nprim(const std::vector<libint2::Shell>& shells) {
  size_t n = 0;
  for (auto shell: shells)
    n = std::max(shell.nprim(), n);
  return n;
}

int max_l(const std::vector<libint2::Shell>& shells) {
  int l = 0;
  for (auto shell: shells)
    for (auto c: shell.contr)
      l = std::max(c.l, l);
  return l;
}

std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell>& shells) {
  std::vector<size_t> result;
  result.reserve(shells.size());

  size_t n = 0;
  for (auto shell: shells) {
    result.push_back(n);
    n += shell.size();
  }

  return result;
}

Matrix compute_1body_ints(const std::vector<libint2::Shell>& shells,
                          libint2::Operator obtype,
                          const std::vector<Atom>& atoms)
{
  using libint2::Shell;
  using libint2::Engine;
  using libint2::Operator;

  const auto n = nbasis(shells);
  Matrix result(n,n);

  // construct the overlap integrals engine
  Engine engine(obtype, max_nprim(shells), max_l(shells), 0);
  // nuclear attraction ints engine needs to know where the charges sit ...
  // the nuclei are charges in this case; in QM/MM there will also be classical charges
  if (obtype == Operator::nuclear) {
    std::vector<std::pair<double,std::array<double,3>>> q;
    for(const auto& atom : atoms) {
      q.push_back( {static_cast<double>(atom.atomic_number), {{atom.x, atom.y, atom.z}}} );
    }
    engine.set_params(q);
  }

  auto shell2bf = map_shell_to_basis_function(shells);

  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto& buf = engine.results();

  // loop over unique shell pairs, {s1,s2} such that s1 >= s2
  // this is due to the permutational symmetry of the real integrals over Hermitian operators: (1|2) = (2|1)
  for(auto s1=0; s1!=shells.size(); ++s1) {

    auto bf1 = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells[s1].size();

    for(auto s2=0; s2<=s1; ++s2) {

      auto bf2 = shell2bf[s2];
      auto n2 = shells[s2].size();

      // compute shell pair; return is the pointer to the buffer
      engine.compute(shells[s1], shells[s2]);

      // "map" buffer to a const Eigen Matrix, and copy it to the corresponding blocks of the result
      Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
      result.block(bf1, bf2, n1, n2) = buf_mat;
      if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1} block, note the transpose!
      result.block(bf2, bf1, n2, n1) = buf_mat.transpose();

    }
  }

  return result;
}

TensorRank4 compute_2body_ints(const std::vector<libint2::Shell>& shells) {

  using libint2::Shell;
  using libint2::Engine;
  using libint2::Operator;

  const auto n = nbasis(shells);
  TensorRank4 g(n,n,n,n);

  // construct the electron repulsion integrals engine
  Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);

  auto shell2bf = map_shell_to_basis_function(shells);

  // buf[0] points to the target shell set after every call  to engine.compute()
  const auto& buf = engine.results();

  // loop over shell pairs of the Fock matrix, {s1,s2}
  // Fock matrix is symmetric, but skipping it here for simplicity (see compute_2body_fock)
  for(auto s1=0; s1!=shells.size(); ++s1) {

    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells[s1].size();

    for(auto s2=0; s2!=shells.size(); ++s2) {

      auto bf2_first = shell2bf[s2];
      auto n2 = shells[s2].size();

      // loop over shell pairs of the density matrix, {s3,s4}
      // again symmetry is not used for simplicity
      for(auto s3=0; s3!=shells.size(); ++s3) {

        auto bf3_first = shell2bf[s3];
        auto n3 = shells[s3].size();

        for(auto s4=0; s4!=shells.size(); ++s4) {

          auto bf4_first = shell2bf[s4];
          auto n4 = shells[s4].size();

          // Coulomb contribution to the Fock matrix is from {s1,s2,s3,s4} integrals
          engine.compute(shells[s1], shells[s2], shells[s3], shells[s4]);
          const auto* buf_1234 = buf[0];
          if (buf_1234 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

          // we don't have an analog of Eigen for tensors (yet ... see github.com/BTAS/BTAS, under development)
          // hence some manual labor here:
          // 1) loop over every integral in the shell set (= nested loops over basis functions in each shell)
          // and 2) add contribution from each integral
          for(auto f1=0, f1234=0; f1!=n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            for(auto f2=0; f2!=n2; ++f2) {
              const auto bf2 = f2 + bf2_first;
              for(auto f3=0; f3!=n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for(auto f4=0; f4!=n4; ++f4, ++f1234) {
                  const auto bf4 = f4 + bf4_first;
                  g(bf1,bf2,bf3,bf4) = buf_1234[f1234];
                }
              }
            }
          }

        }
      }
    }
  }

  return g;
}

