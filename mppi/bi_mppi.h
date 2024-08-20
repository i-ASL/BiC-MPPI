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
    void getNoise(Eigen::MatrixXd &noise, const int &T);
    void solve();
    void dbscan(std::vector<std::vector<int>> &clusters, const Eigen::VectorXd &Di, const Eigen::VectorXd &costs, const int &N, const int &T);
    void calculateU(Eigen::MatrixXd &U, const std::vector<std::vector<int>> &clusters, const Eigen::VectorXd &costs, const Eigen::MatrixXd &Ui, const int &T);
    // void getTrajectory(Eigen::MatrixXd &X, const Eigen::MatrixXd &U, const int &T, const int &clusters_size)
    // void connectControl(Eigen::MatrixXd &Ur, const Eigen::MatrixXd &Xf, const Eigen::MatrixXd &Xb);
    void show();
    
private:
    int Tf;
    int Tb;
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

    Eigen::VectorXd x_init;
    Eigen::VectorXd x_target;
    int Nf;
    int Nb;
    int Nr;
    double gamma_u;
    Eigen::MatrixXd sigma_u;
    double deviation_mu;
    double cost_mu;
    int minpts;
    double epsilon;

    CollisionChecker *collision_checker;


    // Forward
    Eigen::MatrixXd Ui_f;
    Eigen::MatrixXd noise_f;
    Eigen::VectorXd Di_f;
    Eigen::VectorXd costs_f;
    // Eigen::VectorXd weights_f;
    std::vector<std::vector<int>> clusters_f;
    Eigen::MatrixXd Uf;
    Eigen::MatrixXd Xf;

    // Backward
    Eigen::MatrixXd Ui_b;
    Eigen::MatrixXd noise_b;
    Eigen::VectorXd Di_b;
    Eigen::VectorXd costs_b;
    // Eigen::VectorXd weights_f;
    std::vector<std::vector<int>> clusters_b;
    Eigen::MatrixXd Ub;
    Eigen::MatrixXd Xb;

    // Result
    Eigen::MatrixXd Ui_r;
    Eigen::VectorXd costs_R;
    Eigen::MatrixXd xR;

    Eigen::MatrixXd Xr;
    Eigen::MatrixXd Ur;

};

template<typename ModelClass>
BiMPPI::BiMPPI(ModelClass model) {
    this->Tf = model.N;
    this->Tb = model.N;
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
    this->x_init = bi_mppi_param.x_init;
    this->x_target = bi_mppi_param.x_target;
    this->Nf = bi_mppi_param.Nf;
    this->Nb = bi_mppi_param.Nb;
    this->Nr = bi_mppi_param.Nf;
    this->gamma_u = bi_mppi_param.gamma_u;
    this->sigma_u = bi_mppi_param.sigma_u;
    this->deviation_mu = bi_mppi_param.deviation_mu;
    this->cost_mu = bi_mppi_param.cost_mu;
    this->minpts = bi_mppi_param.minpts;
    this->epsilon = bi_mppi_param.epsilon;

    // this->noise_f.resize(dim_u, Tf);
    this->Ui_f.resize(dim_u * Nf, Tf);
    this->Di_f.resize(Nf);
    this->costs_f.resize(Nf);
    // this->weights_f.resize(Nf);

    // this->noise_b.resize(dim_u, Tb);
    this->Ui_b.resize(dim_u * Nb, Tb);
    this->Di_b.resize(Nb);
    this->costs_b.resize(Nb);
    // this->weights_b.resize(Nb);
}

void BiMPPI::setCollisionChecker(CollisionChecker *collision_checker) {
    this->collision_checker = collision_checker;
}

void BiMPPI::getNoise(Eigen::MatrixXd &noise, const int &T) {
    noise = sigma_u * norm_gen.template generate<Eigen::MatrixXd>(dim_u, T, urng);
}

void BiMPPI::solve() {
    // Backward Rollout
    Ui_b = U.replicate(Nb, 1);
    #pragma omp parallel for
    for (int i = 0; i < Nb; ++i) {
        Eigen::MatrixXd Xi(dim_x, Tb + 1);
        getNoise(noise_b, Tb);
        Ui_b.middleRows(i * dim_u, dim_u) += noise_b;
        h(Ui_b.middleRows(i * dim_u, dim_u));

        Xi.col(Tb) = x_target;
        dual2nd cost = 0.0;
        for (int j = Tb - 1; j >= 0; --j) {
            // CHECK
            Xi.col(j) = 2*Xi.col(j+1) - f(Xi.col(j+1), Ui_b.block(i * dim_u, j, dim_u, 1)).cast<double>();
            cost += q(Xi.col(j), Ui_b.block(i * dim_u, j, dim_u, 1));
            if (collision_checker->getCollisionGrid(Xi.col(j))) {
                cost = 1e8;
                break;
            }
        }
        // CHECK
        // cost += 300 * (Xi.col(0) - x_init).squaredNorm();
        costs_b(i) = static_cast<double>(cost.val);
        Di_b(i) = Xi.row(dim_x - 1).mean();
    }
    // std::cout<<"Di_f = "<<Di_f<<std::endl;
    dbscan(clusters_b, Di_b, costs_b, Nb, Tb);
    calculateU(Ub, clusters_b, costs_b, Ui_b, Tb);
    // getTrajectory(Xb, Ub, Tb, clusters_b.size());

    Xb.resize(clusters_b.size() * dim_x, Tb + 1);
    for (int i = 0; i < clusters_b.size(); ++i) {
        Xb.block(i * dim_x, Tb, dim_x, 1) = x_target;
        for (int t = Tb - 1; t >= 0; --t) {
            Xb.block(i * dim_x, t, dim_x, 1) = 2*Xb.block(i * dim_x, t + 1, dim_x, 1) - f(Xb.block(i * dim_x, t + 1, dim_x, 1), Ub.block(i * dim_u, t, dim_u, 1)).cast<double>();
        }
    }

    // Forward Rollout
    Ui_f = U.replicate(Nf, 1);
    #pragma omp parallel for
    for (int i = 0; i < Nf; ++i) {
        Eigen::MatrixXd Xi(dim_x, Tf + 1);
        getNoise(noise_f, Tf);
        Ui_f.middleRows(i * dim_u, dim_u) += noise_f;
        h(Ui_f.middleRows(i * dim_u, dim_u));

        Xi.col(0) = x_init;
        dual2nd cost = 0.0;
        for (int j = 0; j < Tf; ++j) {
            if (collision_checker->getCollisionGrid(Xi.col(j))) {
                cost = 1e8;
                break;
            }
            else {
                cost += q(Xi.col(j), Ui_f.block(i * dim_u, j, dim_u, 1));
                Xi.col(j+1) = f(Xi.col(j), Ui_f.block(i * dim_u, j, dim_u, 1)).cast<double>();
            }
        }
        double min_norm = std::numeric_limits<double>::max();
        for (int cb = 0; cb < clusters_b.size(); ++cb) {
            Eigen::VectorXd xb_col = Xb.block(0, 0, Xi.rows(), 1);
            min_norm = std::min(min_norm, (Xi.colwise() - xb_col).colwise().norm().minCoeff());
        }
        cost += min_norm;
        // cost += p(Xi.col(Tf));

        costs_f(i) = static_cast<double>(cost.val);
        Di_f(i) = Xi.row(dim_x - 1).mean();
    }
    dbscan(clusters_f, Di_f, costs_f, Nf, Tf);
    calculateU(Uf, clusters_f, costs_f, Ui_f, Tf);
    // getTrajectory(Xf, Uf, Tb, clusters_f.size());

    Xf.resize(clusters_f.size() * dim_x, Tf + 1);
    for (int i = 0; i < clusters_f.size(); ++i) {
        Xf.block(i * dim_x, 0, dim_x, 1) = x_init;
        for (int t = 0; t < Tf; ++t) {
            Xf.block(i * dim_x, t + 1, dim_x, 1) = f(Xf.block(i * dim_x, t, dim_x, 1), Uf.block(i * dim_u, t, dim_u, 1)).cast<double>();
        }
    }

    costs_R.resize(clusters_f.size());
    XR.resize(clusters_f.size() * dim_x, );

    Eigen::VectorXd temp(3);
    temp << 10.0, 10.0, 0.01;
    for (int cf = 0; cf < clusters_f.size(); ++cf) {
        double min_norm = std::numeric_limits<double>::max();
        int cb, tf, tb; 
        for (int cb_ = 0; cb_ < clusters_b.size(); ++cb_) {
            for (int tf_ = 0; tf_ < Tf; ++tf_) {
                for (int tb_ = 0; tb_ < Tb; ++tb_) {
                    double norm = (temp.asDiagonal() * (Xf.block(cf * dim_x, tf_, dim_x, 1) - Xb.block(cb_ * dim_x, tb_, dim_x, 1))).norm();
                    if (norm < min_norm) {
                        min_norm = norm;
                        cb = cb_;
                        tf = tf_;
                        tb = tb_;
                    }
                }
            }
        }

        Ur.resize(dim_u, tf + (Tb - tb) - 1);
        Ur << Uf.block(cf * dim_u, 0, dim_u, tf), Ub.block(cb * dim_u, tb + 1, dim_u, (Tb - tb - 1));

        // MPPI whole horizon
        Ui_r = Ur.replicate(Nr, 1);
        int Tr = tf + (Tb - tb) - 1;
        Eigen::VectorXd costs_r(Nr);
        Eigen::VectorXd weights(Nr);
        
        #pragma omp parallel for
        for (int i = 0; i < Nr; ++i) {
            Eigen::MatrixXd Xi(dim_x, Tr + 1);
            getNoise(noise_f, Tr);
            Ui_r.middleRows(i * dim_u, dim_u) += noise_f;
            h(Ui_r.middleRows(i * dim_u, dim_u));

            Xi.col(0) = x_init;
            dual2nd cost = 0.0;
            for (int j = 0; j < Tr; ++j) {
                if (collision_checker->getCollisionGrid(Xi.col(j))) {
                    cost = 1e8;
                    break;
                }
                else {
                    cost += q(Xi.col(j), Ui_r.block(i * dim_u, j, dim_u, 1));

                    if (tf < j) {
                        cost += (Xi.col(j) - Xb.block(cb * dim_x, tb + 1 + (j - tf - 1), dim_x, 1)).norm();
                    }
                    Xi.col(j+1) = f(Xi.col(j), Ui_r.block(i * dim_u, j, dim_u, 1)).cast<double>();
                }
            }
            cost += (Xi.col(Tr) - Xb.block(cb * dim_x, Tb, dim_x, 1)).norm();
            costs_r(i) = static_cast<double>(cost.val);
        }
        double min_cost = costs_r.minCoeff();
        weights = (-gamma_u * (costs_r.array() - min_cost)).exp();
        double total_weight =  weights.sum();
        weights /= total_weight;

        Eigen::MatrixXd Ur = Eigen::MatrixXd::Zero(dim_u, Tr);
        for (int i = 0; i < Nr; ++i) {
            Ur += Ui_r.middleRows(i * dim_u, dim_u) * weights(i);
        }
        h(Ur);

        Eigen::MatrixXd Xi(dim_x, Tr + 1);
        Xi.col(0) = x_init;
        dual2nd cost = 0.0;
        for (int t = 0; t < Tr; ++t) {
            if (collision_checker->getCollisionGrid(Xi.col(j))) {
                cost = 1e8;
                break;
            }
            else {
                cost += q(Xi.col(j), Ui_r.block(i * dim_u, j, dim_u, 1));
                if (tf < j) {
                    cost += (Xi.col(j) - Xb.block(cb * dim_x, tb + 1 + (j - tf - 1), dim_x, 1)).norm();
                }
                Xi.col(j+1) = f(Xi.col(j), Ui_r.block(i * dim_u, j, dim_u, 1)).cast<double>();
            }
            cost += (Xi.col(Tr) - Xb.block(cb * dim_x, Tb, dim_x, 1)).norm();
        }
        Xr.resize(dim_x, Tr + 1);
        Xr.col(0) = x_init;
        for (int t = 0; t < Ur.cols(); ++t) {
            Xr.col(t+1) = f(Xr.col(t), Ur.col(t)).cast<double>();
        }
    }

    // Xr.resize(dim_x, Ur.cols() + 1);
    // Xr.col(0) = x_init;
    // for (int t = 0; t < Ur.cols(); ++t) {
    //     Xr.col(t+1) = f(Xr.col(t), Ur.col(t)).cast<double>();
    // }
}

void BiMPPI::dbscan(std::vector<std::vector<int>> &clusters, const Eigen::VectorXd &Di, const Eigen::VectorXd &costs, const int &N, const int &T) {
    clusters.clear();
    std::vector<bool> core_points(N, false);
    std::map<int, std::vector<int>> core_tree;
    #pragma omp parallel for
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
    }

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        if (minpts < static_cast<int>(core_tree[i].size())) {
            core_points[i] = true;
        }
    }
    
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
}

void BiMPPI::calculateU(Eigen::MatrixXd &U, const std::vector<std::vector<int>> &clusters, const Eigen::VectorXd &costs, const Eigen::MatrixXd &Ui, const int &T) {
    U = Eigen::MatrixXd::Zero(clusters.size() * dim_u, T);
    std::vector<std::vector<double>> weights(clusters.size());
    #pragma omp parallel for
    for (int index = 0; index < clusters.size(); ++index) {
        double min_cost = std::numeric_limits<double>::max();
        for (int k : clusters[index]) {
            min_cost = std::min(min_cost, costs(k));
        }

        weights[index].resize(clusters[index].size());
        for (size_t i = 0; i < clusters[index].size(); ++i) {
            int k = clusters[index][i];
            weights[index][i] = std::exp(-gamma_u * (costs(k) - min_cost));
        }
        double total_weight = std::accumulate(weights[index].begin(), weights[index].end(), 0.0);

        for (size_t i = 0; i < clusters[index].size(); ++i) {
            int k = clusters[index][i];
            U.middleRows(index * dim_u, dim_u) += (weights[index][i] / total_weight) * Ui.middleRows(k * dim_u, dim_u);
        }
    }
}

// void BiMPPI::getTrajectory(Eigen::MatrixXd &X, const Eigen::MatrixXd &U, const int &T, const int &clusters_size) {
// }

// void BiMPPI::connectControl(Eigen::MatrixXd &Ur, const Eigen::MatrixXd &Xf, const Eigen::MatrixXd &Xb) {
// }

void BiMPPI::show() {
    namespace plt = matplotlibcpp;
    plt::subplot(1,2,1);

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
        for (int k : clusters_f[index]) {
            Eigen::MatrixXd Xi(dim_x, Tf+1);
            Xi.col(0) = x_init;
            for (int t = 0; t < Tf; ++t) {
                Xi.col(t+1) = f(Xi.col(t), Ui_f.block(k * dim_u, t, dim_u, 1)).cast<double>();
            }
            std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(Tf));
            for (int i = 0; i < dim_x; ++i) {
                for (int j = 0; j < Tf + 1; ++j) {
                    X_MPPI[i][j] = Xi(i, j);
                }
            }
            // std::cout<<"deviation = "<<Di_f(k)<<"\t";
            // std::cout<<"X = "<<Xi.col(T)(0)<<std::endl;
            std::string color = "C" + std::to_string(index%10);
            // plt::plot(X_MPPI[0], X_MPPI[1], {{"color", color}, {"linewidth", "1.0"}});
        }
    }

    for (int index = 0; index < clusters_b.size(); ++index) {
        for (int k : clusters_b[index]) {
            Eigen::MatrixXd Xi(dim_x, Tb+1);
            Xi.col(Tb) = x_target;
            for (int t = Tb - 1; t >= 0; --t) {
                // Xi.col(t) = Xi.col(t+1) - 0.05*f(Xi.col(t+1), Ui_b.block(k * dim_u, t, dim_u, 1)).cast<double>();
                Xi.col(t) = 2*Xi.col(t+1) - f(Xi.col(t+1), Ui_b.block(k * dim_u, t, dim_u, 1)).cast<double>();
            }
            std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(Tb));
            for (int i = 0; i < dim_x; ++i) {
                for (int j = 0; j < Tb + 1; ++j) {
                    X_MPPI[i][j] = Xi(i, j);
                }
            }
            // std::cout<<"deviation = "<<Di_f(k)<<"\t";
            // std::cout<<"X = "<<Xi.col(T)(0)<<std::endl;
            std::string color = "C" + std::to_string(9 - index%10);
            // plt::plot(X_MPPI[0], X_MPPI[1], {{"color", color}, {"linewidth", "1.0"}});
        }
    }

    for (int index = 0; index < clusters_f.size(); ++index) {
        Eigen::MatrixXd Xi(dim_x, Tf + 1);
        Xi.col(0) = x_init;
        for (int t = 0; t < Tf; ++t) {
            Xi.col(t+1) = f(Xi.col(t), Uf.block(index * dim_u, t, dim_u, 1)).cast<double>();
        }
        std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(Tf));
        for (int i = 0; i < dim_x; ++i) {
            for (int j = 0; j < Tf + 1; ++j) {
                X_MPPI[i][j] = Xi(i, j);
            }
        }
        std::string color = "C" + std::to_string(index%10);
        plt::plot(X_MPPI[0], X_MPPI[1], {{"color", color}, {"linewidth", "10.0"}});
    }

    for (int index = 0; index < clusters_b.size(); ++index) {
        Eigen::MatrixXd Xi(dim_x, Tb + 1);
        Xi.col(Tb) = x_target;
        for (int t = Tb - 1; t >= 0; --t) {
            Xi.col(t) = 2*Xi.col(t+1) - f(Xi.col(t+1), Ub.block(index * dim_u, t, dim_u, 1)).cast<double>();
        }
        std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(Tf));
        for (int i = 0; i < dim_x; ++i) {
            for (int j = 0; j < Tb + 1; ++j) {
                X_MPPI[i][j] = Xi(i, j);
            }
        }
        std::string color = "C" + std::to_string(9 - index%10);
        plt::plot(X_MPPI[0], X_MPPI[1], {{"color", color}, {"linewidth", "10.0"}});
    }
    plt::xlim(0, 3);
    plt::ylim(0, 5);
    plt::grid(true);
    // plt::show();
    
    plt::subplot(1,2,2);
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
    std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(Xr.cols()));
    for (int i = 0; i < dim_x; ++i) {
        for (int j = 0; j < Xr.cols(); ++j) {
            X_MPPI[i][j] = Xr(i, j);
        }
    }
    // std::string color = "C" + std::to_string(9 - index%10);
    plt::plot(X_MPPI[0], X_MPPI[1], {{"color", "black"}, {"linewidth", "10.0"}});

    plt::xlim(0, 3);
    plt::ylim(0, 5);
    plt::grid(true);
    plt::show();
}