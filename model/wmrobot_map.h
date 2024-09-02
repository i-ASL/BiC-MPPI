#pragma once

#include "model_base.h"

class WMRobotMap : public ModelBase {
public:
    WMRobotMap();
    ~WMRobotMap();

    const double velocity = 0.5;
};

WMRobotMap::WMRobotMap() {
    // Dimensions
    dim_x = 3;
    dim_u = 2;

    // Continous Time System
    f = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> Eigen::MatrixXd {
        Eigen::VectorXd x_dot(x.rows());
        x_dot(0) = u(0) * cos(x(2));
        x_dot(1) = u(0) * sin(x(2));
        x_dot(2) = u(1);
        return x_dot;
    };

    // Stage Cost Function
    q = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> double {
        return u.norm();
    };

    // Terminal Cost Function
    p = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& x_target) -> double {
        return (x - x_target).norm();
    };

    h = [&](Eigen::Ref<Eigen::MatrixXd> U) -> void {
        U.row(0) = U.row(0).cwiseMax(velocity).cwiseMin(velocity);
        U.row(1) = U.row(1).cwiseMax(-0.5).cwiseMin(0.5);
        return;
    };
}

WMRobotMap::~WMRobotMap() {
}