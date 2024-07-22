#pragma once

// For align with IPDDP
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

#include "collision_checker.h"
#include "model_base.h"

#include <vector>
#include <iostream>

class MPPI {
public:
    template<typename ModelClass>
    MPPI(ModelClass model);
    ~MPPI();

    void init(int Nu, double lambda, double sigma_u);
    void setCollisionChecker(CollisionChecker *collision_checker);
    void solve();

    Eigen::MatrixXd getInitX();
    Eigen::MatrixXd getInitU();
    Eigen::MatrixXd getResX();
    Eigen::MatrixXd getResU();
    std::vector<double> getAllCost();

    // void randomizeControl();
    // double calculateCost(int i);

private:
    int N;
    int dim_x;
    int dim_u;

    Eigen::MatrixXd X_init;
    Eigen::MatrixXd U_init;
    std::vector<double> all_cost;

    Eigen::MatrixXd X;
    Eigen::MatrixXd U;
    // Discrete Time System
    std::function<VectorXdual2nd(VectorXdual2nd, VectorXdual2nd)> f;
    std::vector<std::function<dual2nd(VectorXdual2nd, VectorXdual2nd)>> fs;
    // Stage Cost Function
    std::function<dual2nd(VectorXdual2nd, VectorXdual2nd)> q;
    // Terminal Cost Function
    std::function<dual2nd(VectorXdual2nd)> p;

    bool is_blocked;

    int Nu;
    double lambda;
    double sigma_u;

    CollisionChecker *collision_checker;

    // Eigen::MatrixXd Xi;
    Eigen::MatrixXd Ui;
    Eigen::VectorXd costs;
    Eigen::VectorXd weights;
};

template<typename ModelClass>
MPPI::MPPI(ModelClass model) {
    this->N = model.N;
    this->dim_x = model.dim_x;
    this->dim_u = model.dim_u;
    
    this->X = model.X;
    this->U = model.U;
    
    this->f = model.f;
    this->fs = model.fs;
    this->q = model.q;
    this->p = model.p;
}

MPPI::~MPPI() {
}

void MPPI::init(int Nu, double lambda, double sigma_u) {
    this->Nu = Nu;
    this->lambda = lambda;
    this->sigma_u = sigma_u;
    this->is_blocked = false;

    this->Ui.resize(dim_u * Nu, N);
    for (int i = 0; i < Nu; ++i) {
        Ui.middleRows(i, dim_u) = U;
    }

    this->costs.resize(Nu);
    this->weights.resize(Nu);

    this->X_init = X;
    this->U_init = U;
}

void MPPI::setCollisionChecker(CollisionChecker *collision_checker) {
    this->is_blocked = true;
    this->collision_checker = collision_checker;
}

void MPPI::solve() {
    Eigen::MatrixXd Xi(dim_x, N+1);
    dual2nd cost;

    for (int i = 0; i < Nu; ++i) {
        Ui.middleRows(i, dim_u) = U + Eigen::MatrixXd::Random(dim_u, N) * this->sigma_u;
        Xi.col(0) = X.col(0);
        cost = 0.0;
        for (int j = 0; j < N; ++j) {
            Xi.col(j+1) = f(Xi.col(j), Ui.block(i, j, dim_u, 1)).cast<double>();
            cost += q(Xi.col(j), Ui.block(i, j, dim_u, 1));
            if (is_blocked) {cost += collision_checker->getCost(Xi.col(j));}
        }
        cost += p(Xi.col(N));
        costs(i) = static_cast<double>(cost.val);
    }

    double min_cost = costs.minCoeff();
    weights = (-lambda * (costs.array() - min_cost)).exp();
    double total_weight =  weights.sum();
    all_cost.push_back(total_weight);
    weights /= total_weight;

    U = Eigen::MatrixXd::Zero(dim_u, N);
    for (int i = 0; i < Nu; ++i) {
        U += Ui.middleRows(i, dim_u) * weights(i);
    }
}

Eigen::MatrixXd MPPI::getInitX() {
    return X_init;
}

Eigen::MatrixXd MPPI::getInitU() {
    return U_init;
}

Eigen::MatrixXd MPPI::getResX() {
    for (int j = 0; j < N; ++j) {
        X.col(j+1) = f(X.col(j), U.col(j)).cast<double>();
    }
    return X;
}

Eigen::MatrixXd MPPI::getResU() {
    return U;
}

std::vector<double> MPPI::getAllCost() {
    return all_cost;
}

// void MPPI::randomizeControl() {
//     U = U + Eigen::MatrixXf::Random(dim_u, Nu) * this->sigma_u;
// }

// double MPPI::calculateCost(Eigen::MatrixXd i) {
//     dual2nd cost = 0.0;
//     for (int j = 0; j < N; ++j) {
//         cost += q(Xi.block(i, j, dim_x, 1), Ui.block(i, j, dim_u, 1)).val;
//     }
//     cost += static_cast<double>(p(Xi.block(i, N, dim_x, 1)).val);
//     return static_cast<double>(cost);
// }