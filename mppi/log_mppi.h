#pragma once

#include "mppi.h"

class LogMPPI : public MPPI {
public:
    template<typename ModelClass>
    LogMPPI(ModelClass model);
    ~LogMPPI();

    Eigen::Rand::LognormalGen<double> log_norm_gen{0.0, 1.0};

    Eigen::MatrixXd getNoise(const int &T) override {
        Eigen::MatrixXd log_distribution = log_norm_gen.template generate<Eigen::MatrixXd>(dim_u, T, urng);
        return (sigma_u * norm_gen.template generate<Eigen::MatrixXd>(dim_u, T, urng)).array() * log_distribution.array();
    }
};

template<typename ModelClass>
LogMPPI::LogMPPI(ModelClass model) : MPPI(model) {
}

LogMPPI::~LogMPPI() {
}

