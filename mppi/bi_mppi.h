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
#include <deque>
#include <map>
#include <iostream>

#include <omp.h>

class BiMPPI {
public:
    template<typename ModelClass>
    BiMPPI(ModelClass model);
    ~BiMPPI();

    void init(BiMPPIParam mppi_param);
    void setCollisionChecker(CollisionChecker *collision_checker);
    Eigen::MatrixXd getNoise();
    void solve();
    std::vector<std::vector<int>> dbscan(const Eigen::VectorXd &Di, const Eigen::VectorXd &costs, const int &N);
    
private:
    int T;
    int dim_x;
    int dim_u;

    Eigen::MatrixXd X_init;
    Eigen::MatrixXd U_init;

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

    int Nf;
    int Nb;
    double gamma_u;
    Eigen::MatrixXd sigma_u;
    double deviation_mu;
    double cost_mu;
    int minpts;
    int epsilon;

    CollisionChecker *collision_checker;

    Eigen::MatrixXd noise;

    Eigen::MatrixXd Ui_f;
    Eigen::VectorXd Di_f;
    Eigen::VectorXd costs_f;
    Eigen::VectorXd weights_f;
};

template<typename ModelClass>
BiMPPI::BiMPPI(ModelClass model) {
    this->T = model.N;
    this->dim_x = model.dim_x;
    this->dim_u = model.dim_u;
    
    this->X = model.X;
    this->U = model.U;
    
    this->f = model.f;
    this->q = model.q;
    this->p = model.p;
    this->h = model.h;
}

BiMPPI::~BiMPPI() {
}

void BiMPPI::init(BiMPPIParam bi_mppi_param) {
    this->Nf = bi_mppi_param.Nf;
    this->Nb = bi_mppi_param.Nb;
    this->gamma_u = bi_mppi_param.gamma_u;
    this->sigma_u = bi_mppi_param.sigma_u;
    this->deviation_mu = bi_mppi_param.deviation_mu;
    this->cost_mu = bi_mppi_param.cost_mu;
    this->minpts = bi_mppi_param.minpts;
    this->epsilon = bi_mppi_param.epsilon;

    this->noise.resize(dim_u, T);
    this->Ui_f.resize(dim_u * Nf, T);
    this->Di_f.resize(Nf);
    this->costs_f.resize(Nf);
    this->weights_f.resize(Nf);
}

void BiMPPI::setCollisionChecker(CollisionChecker *collision_checker) {
    this->collision_checker = collision_checker;
}

Eigen::MatrixXd BiMPPI::getNoise() {
    return sigma_u * norm_gen.template generate<Eigen::MatrixXd>(dim_u, T, urng);
}

void BiMPPI::solve() {
    Ui_f = U.replicate(Nf, 1);
    #pragma omp parallel for
    for (int i = 0; i < Nf; ++i) {
        Eigen::MatrixXd Xi(dim_x, T+1);
        noise = getNoise();
        Di_f(i) = noise.row(1).mean();
        Ui_f.middleRows(i * dim_u, dim_u) += noise;
        h(Ui_f.middleRows(i * dim_u, dim_u));

        Xi.col(0) = X.col(0);
        dual2nd cost = 0.0;
        for (int j = 0; j < T; ++j) {
            if (collision_checker->getCollisionGrid(Xi.col(j))) {
                cost = 1e8;
                break;
            }
            else {
                cost += q(Xi.col(j), Ui_f.block(i * dim_u, j, dim_u, 1));
                Xi.col(j+1) = f(Xi.col(j), Ui_f.block(i * dim_u, j, dim_u, 1)).cast<double>();
            }
        }
        cost += p(Xi.col(T));
        costs_f(i) = static_cast<double>(cost.val);
    }
    std::vector<std::vector<int>> clusters = dbscan(Di_f, costs_f, Nf);

    // U = Eigen::MatrixXd::Zero(dim_u,T);
    // for (int i = 0; i < Nu; ++i) {
    //     U += Ui.middleRows(i * dim_u, dim_u) * weights(i);
    // }
    // h(U);

    // for (int j = 0; j < T; ++j) {
    //     X.col(j+1) = f(X.col(j), U.col(j)).cast<double>();
    // }
}

std::vector<std::vector<int>> BiMPPI::dbscan(const Eigen::VectorXd &Di, const Eigen::VectorXd &costs, const int &N) {
    std::vector<bool> core_points(N, false);
    std::map<int, std::vector<int>> core_tree;
    for (int i = 0; i < N; ++i) {
        std::vector<int> temp_tree;
        for (int j = i + 1; j < N; ++j) {
            if (deviation_mu*(Di.col(i) - Di.col(j)).norm() + cost_mu*(costs.col(i) - costs.col(j)).norm() < epsilon) {
                temp_tree.push_back(j);
            }
        }
        if (minpts < static_cast<int>(temp_tree.size())) {
            core_points[i] = 1;
            core_tree[i] = temp_tree;
        }
    }
    
    std::vector<std::vector<int>> clusters;
    std::vector<bool> visited(N, false);
    for (int i = 0; i < N; ++i) {
        if (!core_points[i]) {continue;}
        if (visited[i]) {continue;}
        std::deque<int> branch;
        std::vector<int> cluster;
        branch.push_back(i);
        cluster.push_back(i);
        visited[i] = true;
        while (!branch.empty()) {
            int now = branch.front();
            branch.pop_front();
            for (int next : core_tree[now]) {
                visited[next] = true;
                cluster.push_back(next);
                if (core_points[next] && !visited[next]) {branch.push_back(next);}
            }
        }
        clusters.push_back(cluster);
    }
    return clusters;
}