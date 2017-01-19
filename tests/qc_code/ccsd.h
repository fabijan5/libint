
double ccsd_energy(const int nbasis, const int ndocc, Eigen::VectorXd &E_orb, TensorRank4 &g);

void ccsd_linear_solver(const int ndocc_so, const int nso, Eigen::MatrixXd &Fso, TensorRank4 &g_so, TensorRank4 &t2);

TensorRank4 get_sigma2(const int ndocc_so, const int nso, Eigen::MatrixXd &Fso, TensorRank4 &g_so, Eigen::MatrixXd &t1, TensorRank4 &t2);

Eigen::MatrixXd get_sigma1(const int ndocc_so, const int nso, Eigen::MatrixXd &Fso, TensorRank4 &g_so, Eigen::MatrixXd &t1, TensorRank4 &t2);

TensorRank4 get_double_amplitude_increment(TensorRank4 &sigma2, Eigen::MatrixXd &Fso, const int ndocc_so, const int nso);

Eigen::MatrixXd get_single_amplitude_increment(Eigen::MatrixXd &sigma1, Eigen::MatrixXd &Fso, const int ndocc_so, const int nso);

TensorRank4 update_double_amplitudes(TensorRank4 &t2, TensorRank4 &dt2, const int ndocc, const int nso);

Eigen::MatrixXd update_single_amplitudes(Eigen::MatrixXd &t1, Eigen::MatrixXd &dt1, const int ndocc, const int nso);

double get_ccsd_energy(const int ndocc_so, const int nso, Eigen::MatrixXd &Fso, Eigen::MatrixXd &t1, TensorRank4 &t2, TensorRank4 &g_so);

double max_abs_sigma2(TensorRank4 &sigma2, const int ndocc_so, const int nso);

double max_abs_sigma1(Eigen::MatrixXd &sigma1, const int ndocc_so, const int nso);

double triples_energy(const int ndocc_so, const int nso, Eigen::MatrixXd &Fso, Eigen::MatrixXd &t1, TensorRank4 &t2, TensorRank4 &g_so);

double lt_triples_energy(const int ndocc_so, const int nso, Eigen::MatrixXd &Fso, Eigen::MatrixXd &t1, TensorRank4 &t2, TensorRank4 &g_so, double E_T_can);

void ccsd_linear_solver_closed_shell(const int ndocc, const int nbasis, Eigen::MatrixXd &F, TensorRank4 &g, TensorRank4 &t2);

TensorRank4 get_sigma2_closed_shell(const int ndocc, const int nbasis, Eigen::MatrixXd &f, TensorRank4 &g, Eigen::MatrixXd &t1, TensorRank4 &t2);

Eigen::MatrixXd get_sigma1_closed_shell(const int ndocc, const int nbasis, Eigen::MatrixXd &F, TensorRank4 &g, Eigen::MatrixXd &t1, TensorRank4 &t2);

double get_ccsd_energy_closed_shell(const int ndocc, const int nbasis, Eigen::MatrixXd &f, Eigen::MatrixXd &t1, TensorRank4 &t2, TensorRank4 &g);

Eigen::MatrixXd get_sigma1_closed_shell_ta(const int ndocc, const int nbasis, Eigen::MatrixXd &F, TensorRank4 &g, Eigen::MatrixXd &t1, TensorRank4 &t2);

TensorRank4 get_sigma2_closed_shell_ta(const int ndocc, const int nbasis, Eigen::MatrixXd &f, TensorRank4 &g, Eigen::MatrixXd &t1, TensorRank4 &t2);

