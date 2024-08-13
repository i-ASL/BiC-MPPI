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