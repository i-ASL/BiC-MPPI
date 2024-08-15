#pragma once
#include "matplotlibcpp.h"

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
    void show();
    
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
    double epsilon;

    CollisionChecker *collision_checker;

    Eigen::MatrixXd noise;

    Eigen::MatrixXd Ui_f;
    Eigen::VectorXd Di_f;
    Eigen::VectorXd costs_f;
    Eigen::VectorXd weights_f;
    std::vector<std::vector<int>> clusters_f;
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
    // #pragma omp parallel for
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
    clusters_f = dbscan(Di_f, costs_f, Nf);

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
    // std::cout<<"costs = "<<costs * cost_mu<<std::endl;
    // std::cout<<"costs.mean() = "<<costs.mean()<<std::endl;
    // std::cout<<"Di = "<<Di * deviation_mu<<std::endl;
    // std::cout<<"Di.mean() = "<<Di.mean()<<std::endl;

    for (int i = 0; i < N; ++i) {
        if (costs(i) > 1E7) {continue;}
        for (int j = i + 1; j < N; ++j) {
            if (costs(j) > 1E7) {continue;}
            // if (deviation_mu * std::abs(Di(i) - Di(j)) + cost_mu * std::abs(costs(i) - costs(j)) < epsilon) {
            if (deviation_mu * std::abs(Di(i) - Di(j)) < epsilon) {
                core_tree[i].push_back(j);
                core_tree[j].push_back(i);
            }
        }
        if (minpts < static_cast<int>(core_tree[i].size())) {
            // std::cout<<"size = "<<static_cast<int>(temp_tree.size())<<std::endl;
            core_points[i] = true;
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
            for (int next : core_tree[now]) {
                if (visited[next]) {continue;}
                visited[next] = true;
                cluster.push_back(next);
                if (core_points[next]) {branch.push_back(next);}
            }
            branch.pop_front();
        }
        clusters.push_back(cluster);
    }
    return clusters;
}

void BiMPPI::show() {
    namespace plt = matplotlibcpp;
    
    // std::vector<std::vector<double>> cost_deviation(Nf, std::vector<double>(Nf));

    // // double min_cost = costs_f.minCoeff();
    // // std::cout<<"min_cost = "<<min_cost<<std::endl;
    // // Eigen::MatrixXd weights = (-gamma_u * (costs_f.array() - min_cost)).exp();
    // // double total_weight =  weights.sum();
    // // weights /= total_weight;
    // // std::cout<<"weights = "<<weights<<std::endl;

    // // costs_f = (costs_f - costs_f.minCoeff()).array().exp();
    // for (int i = 0; i < T; ++i) {
    //     // if (costs_f(i) > 1E7) {continue;}
    //     cost_deviation[0][i] = costs_f(i) * cost_mu;
    //     cost_deviation[1][i] = Di_f(i) * deviation_mu;
    // }
    // plt::scatter(cost_deviation[0], cost_deviation[1], 10.0);
    // plt::xlim(0.0, 1.5);
    // plt::show();

    double resolution = 0.1;
    double hl = resolution / 2;
    for (int i = 0; i < collision_checker->map.size(); ++i) {
        for (int j = 0; j < collision_checker->map[0].size(); ++j) {
            if ((collision_checker->map[i])[j] == 10) {
                double mx = i*resolution;
                double my = j*resolution;
                std::vector<double> oX = {mx-hl, mx+hl, mx+hl, mx-hl, mx-hl};
                std::vector<double> oY = {my-hl,my-hl,my+hl,my+hl,my-hl};
                plt::plot(oX, oY, "k");
            }
        }
    }
    for (int index = 0; index < clusters_f.size(); ++index) {
        std::cout<<clusters_f[index].size()<<std::endl;
        for (int k : clusters_f[index]) {
            std::cout<<"deviation = "<<Di_f(k)<<std::endl;
            Eigen::MatrixXd Xi(dim_x, T+1);
            Xi.col(0) = X.col(0);
            for (int t = 0; t < T; ++t) {
                Xi.col(t+1) = f(Xi.col(t), Ui_f.block(k * dim_u, t, dim_u, 1)).cast<double>();
            }
            std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(T));
            for (int i = 0; i < dim_x; ++i) {
                for (int j = 0; j < T + 1; ++j) {
                    X_MPPI[i][j] = Xi(i, j);
                }
            std::string color = "C" + std::to_string(index%10);
            plt::plot(X_MPPI[0], X_MPPI[1], color);
            }
        }
    }

    plt::xlim(0, 3);
    plt::ylim(0, 5);
    plt::grid(true);
    plt::show();
}