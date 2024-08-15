#include <Eigen/Dense>

struct MPPIParam {
    int Nu;
    double gamma_u;
    Eigen::MatrixXd sigma_u;
};

struct SmoothMPPIParam{
    double dt;
    double lambda;
    Eigen::MatrixXd w;
};

struct BiMPPIParam {
    int Nf;
    int Nb;
    double gamma_u;
    Eigen::MatrixXd sigma_u;
    double deviation_mu;
    double cost_mu;
    int minpts;
    double epsilon;
};