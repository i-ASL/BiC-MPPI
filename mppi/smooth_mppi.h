#pragma once

#include "mppi.h"

class SmoothMPPI : public MPPI {
public:
    template<typename ModelClass>
    SmoothMPPI(ModelClass model);
    ~SmoothMPPI();
    void init2(MPPIParam mppi_pram, SmoothMPPIParam smooth_mppi_param);
    
    Eigen::VectorXd u_dummy;
    Eigen::MatrixXd V;
    Eigen::MatrixXd Vi;
    double dt;
    double lambda;
    Eigen::MatrixXd sigma_u_inv;
    Eigen::MatrixXd w;

    Eigen::VectorXd u_diff;

    void solve() override {
        Vi = V.replicate(Nu, 1);
        Ui = U.replicate(Nu, 1);

        #pragma omp parallel for
        for (int i = 0; i < Nu; ++i) {
            Eigen::MatrixXd noise = getNoise();
            Vi.middleRows(i * dim_u, dim_u) += noise;
            Ui.middleRows(i * dim_u, dim_u) += Vi.middleRows(i * dim_u, dim_u) * dt;
            h(Ui.middleRows(i * dim_u, dim_u));
            Eigen::MatrixXd projected_Vi = (Ui.middleRows(i * dim_u, dim_u) - U) / dt;
            noise = projected_Vi - V;
            Vi.middleRows(i * dim_u, dim_u) = projected_Vi;

            Eigen::MatrixXd Xi(dim_x, N+1);
            Xi.col(0) = X.col(0);
            double cost = 0.0;
            for (int j = 0; j < N; ++j) {
                if (collision_checker->getCollisionGrid(Xi.col(j))) {
                    cost = 1e8;
                    break;
                }
                cost += static_cast<double>(q(Xi.col(j), u_dummy).val) + (gamma_u * (V.col(j).transpose() * sigma_u_inv) * noise.col(j)).sum();
                Xi.col(j+1) = f(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1)).cast<double>();

                if (j == 0) {continue;}
                Eigen::VectorXd u_diff = Ui.block(i * dim_u, j, dim_u, 1) - Ui.block(i * dim_u, j - 1, dim_u, 1);
                cost += u_diff.transpose() * w * u_diff;
                if (j + 1 == N) {cost += static_cast<double>(p(Xi.col(N)).val);}
            }
            costs(i) = cost;
        }

        double min_cost = costs.minCoeff();
        weights = (-(1.0 / gamma_u) * (costs.array() - min_cost)).exp();
        double total_weight =  weights.sum();
        all_cost.push_back(total_weight);
        weights /= total_weight;

        V = Eigen::MatrixXd::Zero(dim_u, N);
        for (int i = 0; i < Nu; ++i) {
            V += Vi.middleRows(i * dim_u, dim_u) * weights(i);
        }

        Eigen::MatrixXd new_U = U + (V * dt);
        h(new_U);
        
        V = (new_U - U) / dt;
        U = new_U;
    }
};

template<typename ModelClass>
SmoothMPPI::SmoothMPPI(ModelClass model) : MPPI(model) {
    u_dummy = Eigen::VectorXd::Zero(dim_u);
    V = Eigen::MatrixXd::Zero(dim_u, N);
    Vi.resize(dim_u * Nu, N);
}

SmoothMPPI::~SmoothMPPI() {
}

void SmoothMPPI::init2(MPPIParam mppi_pram, SmoothMPPIParam smooth_mppi_param) {
    init(mppi_pram);
    sigma_u_inv = sigma_u.inverse();

    dt = smooth_mppi_param.dt;
    lambda = smooth_mppi_param.lambda;
    w = smooth_mppi_param.w;
}
