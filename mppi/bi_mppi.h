#pragma once
#include "matplotlibcpp.h"

#include <EigenRand/EigenRand>

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
    Eigen::MatrixXd getNoise(const int &T);
    void solve();
    void backwardRollout();
    void forwardRollout();
    void selectConnection();
    void concatenate();
    void guideMPPI();

    void dbscan(std::vector<std::vector<int>> &clusters, const Eigen::VectorXd &Di, const Eigen::VectorXd &costs, const int &N, const int &T);
    void calculateU(Eigen::MatrixXd &U, const std::vector<std::vector<int>> &clusters, const Eigen::VectorXd &costs, const Eigen::MatrixXd &Ui, const int &T);
    void show();

    Eigen::MatrixXd U_f0;
    Eigen::MatrixXd U_b0;
    
private:
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

    std::mt19937_64 urng{static_cast<std::uint_fast64_t>(std::time(nullptr))};
    // std::mt19937_64 urng{1};
    Eigen::Rand::NormalGen<double> norm_gen{0.0, 1.0};

    float dt;
    int Tf;
    int Tb;
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
    std::vector<std::vector<int>> clusters_f;
    Eigen::MatrixXd Uf;
    Eigen::MatrixXd Xf;

    // Backward
    std::vector<std::vector<int>> clusters_b;
    Eigen::MatrixXd Ub;
    Eigen::MatrixXd Xb;

    // Connection
    std::vector<std::vector<int>> joints;
    std::vector<Eigen::MatrixXd> Xc;
    std::vector<Eigen::MatrixXd> Uc;

    // Guide
    std::vector<Eigen::MatrixXd> Ur;
    std::vector<double> Cr;
    std::vector<Eigen::MatrixXd> Xr;

    Eigen::MatrixXd Uo;
    Eigen::MatrixXd Xo;
};

template<typename ModelClass>
BiMPPI::BiMPPI(ModelClass model) {
    this->dim_x = model.dim_x;
    this->dim_u = model.dim_u;

    this->f = model.f;
    this->q = model.q;
    this->pt = model.pt;
    this->pi = model.pi;
    this->h = model.h;
}

BiMPPI::~BiMPPI() {
}

void BiMPPI::init(BiMPPIParam bi_mppi_param) {
    this->dt = bi_mppi_param.dt;
    this->Tf = bi_mppi_param.Tf;
    this->Tb = bi_mppi_param.Tb;
    this->x_init = bi_mppi_param.x_init;
    this->x_target = bi_mppi_param.x_target;
    this->Nf = bi_mppi_param.Nf;
    this->Nb = bi_mppi_param.Nb;
    this->Nr = bi_mppi_param.Nr;
    this->gamma_u = bi_mppi_param.gamma_u;
    this->sigma_u = bi_mppi_param.sigma_u;
    this->deviation_mu = bi_mppi_param.deviation_mu;
    this->cost_mu = bi_mppi_param.cost_mu;
    this->minpts = bi_mppi_param.minpts;
    this->epsilon = bi_mppi_param.epsilon;
}

void BiMPPI::setCollisionChecker(CollisionChecker *collision_checker) {
    this->collision_checker = collision_checker;
}

Eigen::MatrixXd BiMPPI::getNoise(const int &T) {
    return sigma_u * norm_gen.template generate<Eigen::MatrixXd>(dim_u, T, urng);
}

void BiMPPI::solve() {
    // omp setting for nested parallel
    omp_set_nested(1);
    
    // 1. Clustered MPPI
    #pragma omp parallel sections
    {
        #pragma omp section
        backwardRollout();

        #pragma omp section
        forwardRollout();
    }
    std::cout<<"Backward : "<<clusters_b.size()<<std::endl;
    std::cout<<"Forward : "<<clusters_f.size()<<std::endl;

    if (clusters_b.empty() || clusters_f.empty()) {
        return;
    }

    // 2. Select Connection
    selectConnection();
    concatenate();

    // 3. Guide MPPI
    guideMPPI();
}

void BiMPPI::backwardRollout() {
    Eigen::MatrixXd Ui = U_b0.replicate(Nb, 1);
    Eigen::VectorXd Di(Nb);
    Eigen::VectorXd costs(Nb);
    #pragma omp parallel for
    for (int i = 0; i < Nb; ++i) {
        Eigen::MatrixXd Xi(dim_x, Tb + 1);
        Eigen::MatrixXd noise = getNoise(Tb);
        Ui.middleRows(i * dim_u, dim_u) += noise;
        h(Ui.middleRows(i * dim_u, dim_u));

        Xi.col(Tb) = x_target;
        double cost = 0.0;
        for (int j = Tb - 1; j >= 0; --j) {
            if (j == Tb - 1) {
                Xi.col(j) = Xi.col(j+1) - (dt * f(Xi.col(j+1), Ui.block(i * dim_u, j, dim_u, 1)));
            }
            else {
                Xi.col(j) = Xi.col(j+1) - (dt * f(Xi.col(j+1), Ui.block(i * dim_u, j + 1, dim_u, 1)));
            }
            cost += q(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1));
            if (collision_checker->getCollisionGrid(Xi.col(j))) {
                cost = 1e8;
                break;
            }
        }
        cost += pi(Xi.col(0), x_init);
        costs(i) = cost;
        Di(i) = Xi.row(dim_x - 1).mean();
    }
    dbscan(clusters_b, Di, costs, Nb, Tb);
    calculateU(Ub, clusters_b, costs, Ui, Tb);

    Xb.resize(clusters_b.size() * dim_x, Tb + 1);
    for (int i = 0; i < clusters_b.size(); ++i) {
        Xb.block(i * dim_x, Tb, dim_x, 1) = x_target;
        for (int t = Tb - 1; t >= 0; --t) {
            if (t == Tb - 1) {
                Xb.block(i * dim_x, t, dim_x, 1) = Xb.block(i * dim_x, t + 1, dim_x, 1) - (dt * f(Xb.block(i * dim_x, t + 1, dim_x, 1), Ub.block(i * dim_u, t, dim_u, 1)));
            }
            else {
                Xb.block(i * dim_x, t, dim_x, 1) = Xb.block(i * dim_x, t + 1, dim_x, 1) - (dt * f(Xb.block(i * dim_x, t + 1, dim_x, 1), Ub.block(i * dim_u, t + 1, dim_u, 1)));
            }
        }
    }
}

void BiMPPI::forwardRollout() {
    Eigen::MatrixXd Ui = U_f0.replicate(Nf, 1);
    Eigen::VectorXd Di(Nf);
    Eigen::VectorXd costs(Nf);
    #pragma omp parallel for
    for (int i = 0; i < Nf; ++i) {
        Eigen::MatrixXd Xi(dim_x, Tf + 1);
        Eigen::MatrixXd noise = getNoise(Tf);
        Ui.middleRows(i * dim_u, dim_u) += noise;
        h(Ui.middleRows(i * dim_u, dim_u));

        Xi.col(0) = x_init;
        double cost = 0.0;
        for (int j = 0; j < Tf; ++j) {
            if (collision_checker->getCollisionGrid(Xi.col(j))) {
                cost = 1e8;
                break;
            }
            else {
                cost += q(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1));
                Xi.col(j+1) = Xi.col(j) + (dt * f(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1)));
            }
        }
        cost += pt(Xi.col(Tf), x_target);
        costs(i) = cost;
        Di(i) = Xi.row(dim_x - 1).mean();
    }
    dbscan(clusters_f, Di, costs, Nf, Tf);
    calculateU(Uf, clusters_f, costs, Ui, Tf);

    Xf.resize(clusters_f.size() * dim_x, Tf + 1);
    for (int i = 0; i < clusters_f.size(); ++i) {
        Xf.block(i * dim_x, 0, dim_x, 1) = x_init;
        for (int t = 0; t < Tf; ++t) {
            Xf.block(i * dim_x, t + 1, dim_x, 1) = Xf.block(i * dim_x, t, dim_x, 1) + (dt * f(Xf.block(i * dim_x, t, dim_x, 1), Uf.block(i * dim_u, t, dim_u, 1)));
        }
    }
}

void BiMPPI::selectConnection() {
    joints.clear();
    for (int cf = 0; cf < clusters_f.size(); ++cf) {
        double min_norm = std::numeric_limits<double>::max();
        int cb, df, db;
        for (int cb_ = 0; cb_ < clusters_b.size(); ++cb_) {
            for (int df_ = 0; df_ < Tf; ++df_) {
                for (int db_ = 0; db_ < Tb; ++db_) {
                    double norm = (Xf.block(cf * dim_x, df_, dim_x, 1) - Xb.block(cb_ * dim_x, db_, dim_x, 1)).norm();
                    if (norm < min_norm) {
                        min_norm = norm;
                        cb = cb_;
                        df = df_;
                        db = db_;
                    }
                }
            }
        }
        joints.push_back({cf, cb, df, db});
    }
}

void BiMPPI::concatenate() {
    Uc.clear();
    Xc.clear();
    for (std::vector<int> joint : joints) {
        int cf = joint[0];
        int cb = joint[1];
        int df = joint[2];
        int db = joint[3];

        Eigen::MatrixXd U(dim_u, df + (Tb - db) - 1);
        Eigen::MatrixXd X(dim_x, df + (Tb - db));

        // Exception Handling
        U << Uf.block(cf * dim_u, 0, dim_u, df), Ub.block(cb * dim_u, db + 1, dim_u, Tb - db - 1);
        X << Xf.block(cf * dim_x, 0, dim_x, df), Xb.block(cb * dim_x, db + 1, dim_x, Tb - db);

        Uc.push_back(U);
        Xc.push_back(X);
    }
}

void BiMPPI::guideMPPI() {
    if (joints.empty()) {
        std::cout<<"Joint Empty"<<std::endl;
        return;
    }

    Ur.clear();
    Cr.clear();
    Xr.clear();
    
    #pragma omp parallel for
    for (int r = 0; r < joints.size(); ++r) {
        Eigen::MatrixXd Ui = Uc[r].replicate(Nr, 1);
        Eigen::MatrixXd X_res = Xc[r];
        int Tr = Uc[r].cols();
        Eigen::VectorXd costs_r(Nr);
        Eigen::VectorXd weights(Nr);

        #pragma omp parallel for
        for (int i = 0; i < Nr; ++i) {
            Eigen::MatrixXd Xi(dim_x, Tr + 1);
            Eigen::MatrixXd noise = getNoise(Tr);
            Ui.middleRows(i * dim_u, dim_u) += noise;
            h(Ui.middleRows(i * dim_u, dim_u));

            Xi.col(0) = x_init;
            double cost = 0.0;
            for (int j = 0; j < Tr; ++j) {
                cost += q(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1));
                Xi.col(j+1) = Xi.col(j) + (dt * f(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1)));
                if (collision_checker->getCollisionGrid(Xi.col(j))) {
                    cost = 1e8;
                    // break;
                }
            }
            // Guide Cost
            cost += pt(Xi.col(Tr), x_target);
            cost += 1000 * (Xi - X_res).colwise().norm().sum();
            costs_r(i) = cost;
        }
        double min_cost = costs_r.minCoeff();
        weights = (-gamma_u * (costs_r.array() - min_cost)).exp();
        double total_weight =  weights.sum();
        weights /= total_weight;

        Eigen::MatrixXd Ures = Eigen::MatrixXd::Zero(dim_u, Tr);
        for (int i = 0; i < Nr; ++i) {
            Ures += Ui.middleRows(i * dim_u, dim_u) * weights(i);
        }
        h(Ures);

        // OCP Cost Calculation
        Eigen::MatrixXd Xi(dim_x, Tr + 1);
        Xi.col(0) = x_init;
        double cost = 0.0;
        for (int j = 0; j < Tr; ++j) {
            if (collision_checker->getCollisionGrid(Xi.col(j))) {
                cost = 1e8;
                break;
            }
            else {
                cost += q(Xi.col(j), Ures.col(j));
                Xi.col(j+1) = Xi.col(j) + (dt * f(Xi.col(j), Ures.col(j)));
            }
        }
        cost += pt(Xi.col(Tr), x_target);

        Ur.push_back(Ures);
        Cr.push_back(cost);
        Xr.push_back(Xi);
    }

    // Optimal Control Result
    double min_cost = std::numeric_limits<double>::max();
    int index = 0;
    for (int r = 0; r < joints.size(); ++r) {
        if (Cr[r] < min_cost) {
            min_cost = Cr[r];
            index = r;
        }
    }

    Uo = Ur[index];
    Xo = Xr[index];
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
            if (deviation_mu * std::abs(Di(i) - Di(j)) < epsilon) {
                #pragma omp critical
                {
                core_tree[i].push_back(j);
                core_tree[j].push_back(i);
                }
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        if (minpts < core_tree[i].size()) {
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
    #pragma omp parallel for
    for (int index = 0; index < clusters.size(); ++index) {
        int pts = clusters[index].size();
        std::vector<double> weights(pts);
        double min_cost = std::numeric_limits<double>::max();
        for (int k : clusters[index]) {
            min_cost = std::min(min_cost, costs(k));
        }

        for (size_t i = 0; i < pts; ++i) {
            weights[i] = std::exp(-gamma_u * (costs(clusters[index][i]) - min_cost));
        }
        double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);

        for (size_t i = 0; i < pts; ++i) {
            U.middleRows(index * dim_u, dim_u) += (weights[i] / total_weight) * Ui.middleRows(clusters[index][i] * dim_u, dim_u);
        }
    }
}

void BiMPPI::show() {
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

    // for (int index = 0; index < clusters_f.size(); ++index) {
    //     for (int k : clusters_f[index]) {
    //         Eigen::MatrixXd Xi(dim_x, Tf+1);
    //         Xi.col(0) = x_init;
    //         for (int t = 0; t < Tf; ++t) {
    //             Xi.col(t+1) = f(Xi.col(t), Ui_f.block(k * dim_u, t, dim_u, 1)).cast<double>();
    //         }
    //         std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(Tf));
    //         for (int i = 0; i < dim_x; ++i) {
    //             for (int j = 0; j < Tf + 1; ++j) {
    //                 X_MPPI[i][j] = Xi(i, j);
    //             }
    //         }
    //         // std::cout<<"deviation = "<<Di_f(k)<<"\t";
    //         // std::cout<<"X = "<<Xi.col(T)(0)<<std::endl;
    //         std::string color = "C" + std::to_string(index%10);
    //         // plt::plot(X_MPPI[0], X_MPPI[1], {{"color", color}, {"linewidth", "1.0"}});
    //     }
    // }

    // for (int index = 0; index < clusters_b.size(); ++index) {
    //     for (int k : clusters_b[index]) {
    //         Eigen::MatrixXd Xi(dim_x, Tb+1);
    //         Xi.col(Tb) = x_target;
    //         for (int t = Tb - 1; t >= 0; --t) {
    //             // Xi.col(t) = Xi.col(t+1) - 0.05*f(Xi.col(t+1), Ui_b.block(k * dim_u, t, dim_u, 1)).cast<double>();
    //             Xi.col(t) = 2*Xi.col(t+1) - f(Xi.col(t+1), Ui_b.block(k * dim_u, t, dim_u, 1)).cast<double>();
    //         }
    //         std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(Tb));
    //         for (int i = 0; i < dim_x; ++i) {
    //             for (int j = 0; j < Tb + 1; ++j) {
    //                 X_MPPI[i][j] = Xi(i, j);
    //             }
    //         }
    //         // std::cout<<"deviation = "<<Di_f(k)<<"\t";
    //         // std::cout<<"X = "<<Xi.col(T)(0)<<std::endl;
    //         std::string color = "C" + std::to_string(9 - index%10);
    //         // plt::plot(X_MPPI[0], X_MPPI[1], {{"color", color}, {"linewidth", "1.0"}});
    //     }
    // }

    for (int index = 0; index < clusters_f.size(); ++index) {
        std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(Tf));
        for (int i = 0; i < dim_x; ++i) {
            for (int j = 0; j < Tf + 1; ++j) {
                X_MPPI[i][j] = Xf(index * dim_x + i, j);
            }
        }
        std::string color = "C" + std::to_string(index%10);
        plt::plot(X_MPPI[0], X_MPPI[1], {{"color", color}, {"linewidth", "10.0"}});
    }

    for (int index = 0; index < clusters_b.size(); ++index) {
        std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(Tb));
        for (int i = 0; i < dim_x; ++i) {
            for (int j = 0; j < Tb + 1; ++j) {
                X_MPPI[i][j] = Xb(index * dim_x + i, j);
            }
        }
        std::string color = "C" + std::to_string(index%10);
        plt::plot(X_MPPI[0], X_MPPI[1], {{"color", color}, {"linewidth", "10.0"}});
    }

    // // plt::xlim(0, 3);
    // // plt::ylim(0, 5);
    // // plt::grid(true);
    // // // plt::show();
    
    // plt::subplot(1,2,2);
    // for (int i = 0; i < collision_checker->map.size(); ++i) {
    //     for (int j = 0; j < collision_checker->map[0].size(); ++j) {
    //         if ((collision_checker->map[i])[j] == 10) {
    //             double mx = i*resolution;
    //             double my = j*resolution;
    //             std::vector<double> oX = {mx-hl, mx+hl, mx+hl, mx-hl, mx-hl};
    //             std::vector<double> oY = {my-hl,my-hl,my+hl,my+hl,my-hl};
    //             plt::plot(oX, oY, "k");
    //         }
    //     }
    // }
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