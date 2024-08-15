#include <bi_wmrobot_map.h>
#include <bi_mppi.h>

#include <omp.h>
#include <EigenRand/EigenRand>

#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <ctime>


int main() {
    auto model = BiWMRobotMap();

    BiMPPIParam bi_mppi_param;
    bi_mppi_param.Nf = 1000;
    bi_mppi_param.Nb = 1000;
    bi_mppi_param.gamma_u = 100;
    Eigen::VectorXd sigma_u(model.dim_u);
    sigma_u << 0.0, 0.5;
    bi_mppi_param.sigma_u = sigma_u.asDiagonal();
    bi_mppi_param.deviation_mu = 1.0;
    bi_mppi_param.cost_mu = 1.0;
    bi_mppi_param.minpts = 5;
    bi_mppi_param.epsilon = 1.0;

    BiMPPI bi_mppi(model);
    bi_mppi.init(bi_mppi_param);
    bi_mppi.solve();

    return 0;
}
