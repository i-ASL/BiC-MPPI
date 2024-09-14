#pragma once

#include "model_base.h"
#include <iostream>

class Quadrotor : public ModelBase {
public:
    Quadrotor();
    ~Quadrotor();
    
    const double g = 9.81;
};

Quadrotor::Quadrotor() {
    // Dimensions
    dim_x = 6;
    dim_u = 3;

    // Continous Time System
    f = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> Eigen::MatrixXd {
        Eigen::VectorXd x_dot(x.rows());
        x_dot(0) = x(3);
        x_dot(1) = x(4);
        x_dot(2) = x(5);
        x_dot(3) = u(0);
        x_dot(4) = u(1);
        x_dot(5) = u(2) - g;
        return x_dot;
    };

    // Stage Cost Function
    q = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> double {
        return u.norm();
    };

    // Terminal Cost Function
    p = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& x_target) -> double {
        return (x.head(3) - x_target.head(3)).norm();
    };

    h = [&](Eigen::Ref<Eigen::MatrixXd> U) -> void {
        Eigen::VectorXd norms = U.colwise().norm();
        Eigen::VectorXd normMask = (norms.array() >= 20.0).select(1.0 / norms.array(), 1.0);
        // U = U.colwise() * normMask.array();
        for (int i = 0; i < U.cols(); ++i) {
            U.col(i).array() *= normMask(i);
        }

        Eigen::VectorXd V = U.topRows(2).colwise().norm();
        Eigen::VectorXd S = U.row(2).array() * std::tan(M_PI / 3.0);
        // U = (V.array() < -S.array()).select(Eigen::MatrixXd::Zero(dim_u, U.cols()), U);
        for (int i = 0; i < U.cols(); ++i) {
            if (V(i) < -S(i)) {
                U.col(i).setZero();
                V(i) = 0;
                S(i) = 0;
            }
        }

        // V = U.topRows(2).colwise().norm();
        // S = U.row(2).array() * std::tan(M_PI / 3.0);
        Eigen::VectorXd muls = 0.5 * (1 + S.array() / V.array());
        Eigen::MatrixXd UV(dim_u, U.cols());
        UV.topRows(2) = U.topRows(2);
        UV.row(2) = V;
        // U = (norms.array() > S.array().abs()).select(updated_values, U);
        for (int i = 0; i < U.cols(); ++i) {
            if (V(i) > std::abs(S(i))) {
                U.col(i) = UV.col(i) * muls(i);
            }
        }
        return;
    };
}

Quadrotor::~Quadrotor() {
}