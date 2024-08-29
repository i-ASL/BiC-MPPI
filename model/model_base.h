#pragma once

#include <Eigen/Dense>

class ModelBase {
public:
    ModelBase();
    ~ModelBase();

    int dim_x;
    int dim_u;

    // Discrete Time System
    std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> f;
    // Stage Cost Function
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
    // Terminal Cost Function
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> pt;
    // Initial Cost Function
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> pi;
    // Projection
    std::function<void(Eigen::Ref<Eigen::MatrixXd>)> h;
};

ModelBase::ModelBase() {
};

ModelBase::~ModelBase() {
};


