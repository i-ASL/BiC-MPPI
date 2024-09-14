#pragma once
#include "matplotlibcpp.h"

#include <EigenRand/EigenRand>

#include "mppi_param.h"
#include "collision_checker.h"
#include "model_base.h"

#include <ctime>
#include <vector>
#include <chrono>
#include <iostream>

#include <omp.h>

class MPPI {
public:
    template<typename ModelClass>
    MPPI(ModelClass model);
    ~MPPI();

    void init(MPPIParam mppi_param);
    void setCollisionChecker(CollisionChecker *collision_checker);
    virtual Eigen::MatrixXd getNoise(const int &T);
    void move();
    virtual void solve();
    void show();
    void showTraj();

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> finish;
    std::chrono::duration<double> elapsed_1;
    double elapsed;

    Eigen::MatrixXd U_0;
    Eigen::VectorXd x_init;
    Eigen::VectorXd x_target;
    Eigen::MatrixXd Uo;
    Eigen::MatrixXd Xo;

protected:
    int dim_x;
    int dim_u;

    // Discrete Time System
    std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> f;
    // Stage Cost Function
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
    // Terminal Cost Function
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> p;
    // Projection
    std::function<void(Eigen::Ref<Eigen::MatrixXd>)> h;

    std::mt19937_64 urng{static_cast<std::uint_fast64_t>(std::time(nullptr))};
    // std::mt19937_64 urng{1};
    Eigen::Rand::NormalGen<double> norm_gen{0.0, 1.0};

    // Parameters
    float dt;
    int T;
    int N;
    double gamma_u;
    Eigen::MatrixXd sigma_u;
    
    CollisionChecker *collision_checker;

    Eigen::VectorXd u0;

    std::vector<Eigen::MatrixXd> visual_traj;
};

template<typename ModelClass>
MPPI::MPPI(ModelClass model) {
    this->dim_x = model.dim_x;
    this->dim_u = model.dim_u;

    this->f = model.f;
    this->q = model.q;
    this->p = model.p;
    this->h = model.h;
}

MPPI::~MPPI() {
}

void MPPI::init(MPPIParam mppi_param) {
    this->dt = mppi_param.dt;
    this->T = mppi_param.T;
    this->x_init = mppi_param.x_init;
    this->x_target = mppi_param.x_target;
    this->N = mppi_param.N;
    this->gamma_u = mppi_param.gamma_u;
    this->sigma_u = mppi_param.sigma_u;

    u0 = Eigen::VectorXd::Zero(dim_u);
    Xo = Eigen::MatrixXd::Zero(dim_x, T+1);
}

void MPPI::setCollisionChecker(CollisionChecker *collision_checker) {
    this->collision_checker = collision_checker;
}

Eigen::MatrixXd MPPI::getNoise(const int &T) {
    return sigma_u * norm_gen.template generate<Eigen::MatrixXd>(dim_u, T, urng);
}

void MPPI::move() {
    x_init = x_init + (dt * f(x_init, u0));
    U_0.leftCols(T-1) = Uo.rightCols(T-1);
}

void MPPI::solve() {
    start = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXd Ui = U_0.replicate(N, 1);
    Eigen::VectorXd costs(N);
    Eigen::VectorXd weights(N);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        Eigen::MatrixXd Xi(dim_x, T+1);
        Eigen::MatrixXd noise = getNoise(T);
        Ui.middleRows(i * dim_u, dim_u) += noise;
        h(Ui.middleRows(i * dim_u, dim_u));

        Xi.col(0) = x_init;
        double cost = 0.0;
        for (int j = 0; j < T; ++j) {
            cost += p(Xi.col(j), x_target);
            // cost += q(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1));
            Xi.col(j+1) = Xi.col(j) + (dt * f(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1)));
        }

        cost += p(Xi.col(T), x_target);
        for (int j = 1; j < T+1; ++j) {
            if (collision_checker->getCollisionGrid(Xi.col(j))) {
                cost = 1e8;
                break;
            }
        }
        costs(i) = cost;
    }

    double min_cost = costs.minCoeff();
    weights = (-gamma_u * (costs.array() - min_cost)).exp();
    double total_weight =  weights.sum();
    weights /= total_weight;

    Uo = Eigen::MatrixXd::Zero(dim_u, T);
    for (int i = 0; i < N; ++i) {
        Uo += Ui.middleRows(i * dim_u, dim_u) * weights(i);
    }
    h(Uo);


    finish = std::chrono::high_resolution_clock::now();
    elapsed_1 = finish - start;
    // std::cout<<elapsed_1.count()<<std::endl;

    elapsed = elapsed_1.count();

    u0 = Uo.col(0);

    Xo.col(0) = x_init;
    for (int j = 0; j < T; ++j) {
        Xo.col(j+1) = Xo.col(j) + (dt * f(Xo.col(j), Uo.col(j)));
    }

    visual_traj.push_back(x_init);
}


void MPPI::show() {
    namespace plt = matplotlibcpp;
    // plt::subplot(1,2,1);

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

    std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(Xo.cols()));
    for (int i = 0; i < dim_x; ++i) {
        for (int j = 0; j < Xo.cols(); ++j) {
            X_MPPI[i][j] = Xo(i, j);
        }
    }
    // std::string color = "C" + std::to_string(9 - index%10);
    plt::plot(X_MPPI[0], X_MPPI[1], {{"color", "black"}, {"linewidth", "10.0"}});

    plt::xlim(0, 3);
    plt::ylim(0, 5);
    plt::grid(true);
    plt::show();
}

void MPPI::showTraj() {
    namespace plt = matplotlibcpp;

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

    std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(visual_traj.size()));
    for (int i = 0; i < dim_x; ++i) {
        for (int j = 0; j < visual_traj.size(); ++j) {
            X_MPPI[i][j] = visual_traj[j](i);
        }
    }
    // std::string color = "C" + std::to_string(9 - index%10);
    plt::plot(X_MPPI[0], X_MPPI[1], {{"color", "black"}, {"linewidth", "10.0"}});

    plt::xlim(0, 3);
    plt::ylim(0, 5);
    plt::grid(true);
    plt::show();
}