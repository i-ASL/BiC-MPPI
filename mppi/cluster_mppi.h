#pragma once

#include "mppi.h"

#include <deque>
#include <map>

class ClusterMPPI : public MPPI {
public:
    template<typename ModelClass>
    ClusterMPPI(ModelClass model);
    ~ClusterMPPI();

    Eigen::MatrixXd U;
    Eigen::MatrixXd X;

    void dbscan(std::vector<std::vector<int>> &clusters, const Eigen::MatrixXd &Di, const Eigen::VectorXd &costs, const int &N, const int &T);
    void calculateU(Eigen::MatrixXd &U, const std::vector<std::vector<int>> &clusters, const Eigen::VectorXd &costs, const Eigen::MatrixXd &Ui, const int &T);


    void solve() override {
        std::vector<int> full_cluster(N);
        std::iota(full_cluster.begin(), full_cluster.end(), 0);

        start = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXd Ui = U_0.replicate(N, 1);
        Eigen::MatrixXd Di(dim_u, N);
        Eigen::VectorXd costs(N);
        Eigen::VectorXd weights(N);
        bool all_feasible = true;
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
                    all_feasible = false;
                    break;
                }
            }
            costs(i) = cost;
            Di.col(i) = (Ui.middleRows(i * dim_u, dim_u) - U_0).rowwise().mean();
        }

        std::vector<std::vector<int>> clusters;

        if (!all_feasible) {dbscan(clusters, Di, costs, N, T);}
        else {clusters.clear();}

        if (clusters.empty()) {clusters.push_back(full_cluster);}
        calculateU(U, clusters, costs, Ui, T);

        double min_cost = std::numeric_limits<double>::max();
        int min_index = 0;
        
        for (int i = 0; i < clusters.size(); ++i) {
            Eigen::MatrixXd Xi(dim_x, T+1);
            Xi.col(0) = x_init;
            double cost = 0.0;
            for (int j = 0; j < T; ++j) {
                cost += p(Xi.col(j), x_target);
                Xi.col(j+1) = Xi.col(j) + (dt * f(Xi.col(j), U.block(i * dim_u, j, dim_u, 1)));
            }
            cost += p(Xi.col(T), x_target);
            for (int j = 1; j < T+1; ++j) {
                if (collision_checker->getCollisionGrid(Xi.col(j))) {
                    cost = 1e8;
                    break;
                }
            }
            if (cost < min_cost) {
                min_cost = cost;
                min_index = i;
            }
        }

        Uo = U.middleRows(min_index * dim_u, dim_u);

        finish = std::chrono::high_resolution_clock::now();
        elapsed_1 = finish - start;
        elapsed = elapsed_1.count();

        u0 = Uo.col(0);

        Xo.col(0) = x_init;
        for (int j = 0; j < T; ++j) {
            Xo.col(j+1) = Xo.col(j) + (dt * f(Xo.col(j), Uo.col(j)));
        }

        visual_traj.push_back(x_init);
    }

private:
    // Parameters
    double deviation_mu;
    double cost_mu;
    int minpts;
    double epsilon;
    double psi;
};

template<typename ModelClass>
ClusterMPPI::ClusterMPPI(ModelClass model) : MPPI(model) {
    deviation_mu = 1.0;
    cost_mu = 1.0;
    minpts = 5;
    epsilon = 0.01;
    psi = 0.6;
}

ClusterMPPI::~ClusterMPPI() {
}

void ClusterMPPI::dbscan(std::vector<std::vector<int>> &clusters, const Eigen::MatrixXd &Di, const Eigen::VectorXd &costs, const int &N, const int &T) {
    clusters.clear();
    std::vector<bool> core_points(N, false);
    std::map<int, std::vector<int>> core_tree;

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        if (costs(i) > 1E7) {continue;}
        for (int j = i + 1; j < N; ++j) {
            if (costs(j) > 1E7) {continue;}
            if (deviation_mu * (Di.col(i) - Di.col(j)).norm() < epsilon) {
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

void ClusterMPPI::calculateU(Eigen::MatrixXd &U, const std::vector<std::vector<int>> &clusters, const Eigen::VectorXd &costs, const Eigen::MatrixXd &Ui, const int &T) {
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
        h(U.middleRows(index * dim_u, dim_u));
    }
}