#pragma once

#include <EigenRand/EigenRand>

// For align with IPDDP
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

#include "mppi_param.h"
#include "collision_checker.h"
#include "model_base.h"

#include <ctime>
#include <vector>
#include <iostream>

#include <omp.h>

class MPPI {
public:
    template<typename ModelClass>
    MPPI(ModelClass model);
    ~MPPI();

    void init(MPPIParam mppi_param);
    void setCollisionChecker(CollisionChecker *collision_checker);
    virtual Eigen::MatrixXd getNoise();
    virtual void solve();
    void solve(Eigen::MatrixXd &X, Eigen::MatrixXd &U);
    
    Eigen::MatrixXd getInitX();
    Eigen::MatrixXd getInitU();
    Eigen::MatrixXd getResX();
    Eigen::MatrixXd getResU();
    std::vector<double> getAllCost();
    
protected:
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
    // Stage Cost Function
    std::function<dual2nd(VectorXdual2nd, VectorXdual2nd)> q;
    // Terminal Cost Function
    std::function<dual2nd(VectorXdual2nd)> p;
    
    std::function<void(Eigen::Ref<Eigen::MatrixXd>)> h;

    std::mt19937_64 urng{static_cast<std::uint_fast64_t>(std::time(nullptr))};
    // std::mt19937_64 urng{1};
    Eigen::Rand::NormalGen<double> norm_gen{0.0, 1.0};

    int Nu;
    double gamma_u;
    Eigen::MatrixXd sigma_u;

    CollisionChecker *collision_checker;

    Eigen::MatrixXd noise;

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
    this->q = model.q;
    this->p = model.p;
    this->h = model.h;
}

MPPI::~MPPI() {
}

void MPPI::init(MPPIParam mppi_param) {
    this->Nu = mppi_param.Nu;
    this->gamma_u = mppi_param.gamma_u;
    this->sigma_u = mppi_param.sigma_u;

    this->noise.resize(dim_u, N);
    this->Ui.resize(dim_u * Nu, N);
    this->costs.resize(Nu);
    this->weights.resize(Nu);

    // this->X_init = X;
    // this->U_init = U;
}

void MPPI::setCollisionChecker(CollisionChecker *collision_checker) {
    this->collision_checker = collision_checker;
}

Eigen::MatrixXd MPPI::getNoise() {
    return sigma_u * norm_gen.template generate<Eigen::MatrixXd>(dim_u, N, urng);
}

void MPPI::solve() {
    Ui = U.replicate(Nu, 1);
    #pragma omp parallel for
    for (int i = 0; i < Nu; ++i) {
        Eigen::MatrixXd Xi(dim_x, N+1);
        Eigen::MatrixXd noise = getNoise();
        Ui.middleRows(i * dim_u, dim_u) += noise;
        h(Ui.middleRows(i * dim_u, dim_u));

        Xi.col(0) = X.col(0);
        dual2nd cost = 0.0;
        for (int j = 0; j < N; ++j) {
            if (collision_checker->getCollisionGrid(Xi.col(j))) {
                cost = 1e8;
                break;
            }
            else {
                cost += q(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1));
                Xi.col(j+1) = f(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1)).cast<double>();
            }
        }
        cost += p(Xi.col(N));
        costs(i) = static_cast<double>(cost.val);
    }

    double min_cost = costs.minCoeff();
    weights = (-gamma_u * (costs.array() - min_cost)).exp();
    double total_weight =  weights.sum();
    all_cost.push_back(total_weight);
    weights /= total_weight;

    U = Eigen::MatrixXd::Zero(dim_u, N);
    for (int i = 0; i < Nu; ++i) {
        U += Ui.middleRows(i * dim_u, dim_u) * weights(i);
    }
    h(U);

    // for (int j = 0; j < N; ++j) {
    //     X.col(j+1) = f(X.col(j), U.col(j)).cast<double>();
    // }
}

void MPPI::solve(Eigen::MatrixXd &X, Eigen::MatrixXd &U) {
    Ui = U.replicate(Nu, 1);

    #pragma omp parallel for
    for (int i = 0; i < Nu; ++i) {
        Eigen::MatrixXd Xi(dim_x, N+1);
        Eigen::MatrixXd noise = getNoise();
        Ui.middleRows(i * dim_u, dim_u) += noise;
        h(Ui.middleRows(i * dim_u, dim_u));

        Xi.col(0) = X.col(0);
        dual2nd cost = 0.0;
        for (int j = 0; j < N; ++j) {
            if (collision_checker->getCollisionGrid(Xi.col(j))) {
                cost = 1e8;
                break;
            }
            else {
                cost += q(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1));
                Xi.col(j+1) = f(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1)).cast<double>();
            }
        }
        cost += p(Xi.col(N));
        costs(i) = static_cast<double>(cost.val);
    }

    double min_cost = costs.minCoeff();
    weights = (-gamma_u * (costs.array() - min_cost)).exp();
    double total_weight =  weights.sum();
    all_cost.push_back(total_weight);
    weights /= total_weight;

    U = Eigen::MatrixXd::Zero(dim_u, N);
    for (int i = 0; i < Nu; ++i) {
        U += Ui.middleRows(i * dim_u, dim_u) * weights(i);
    }
    h(U);

    for (int j = 0; j < N; ++j) {
        X.col(j+1) = f(X.col(j), U.col(j)).cast<double>();
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