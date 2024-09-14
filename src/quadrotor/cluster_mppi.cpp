#include <quadrotor.h>
#include <cluster_mppi.h>

#include <iostream>
#include <Eigen/Dense>
#include <chrono>

int main() {
    auto model = Quadrotor();
    
    using Solver = ClusterMPPI;
    using SolverParam = MPPIParam;

    SolverParam param;
    param.dt = 0.1;
    param.T = 100;
    param.x_init.resize(model.dim_x);
    param.x_init << 1.5, 0.0, 5.0, 0.0, 0.0, 0.0;
    param.x_target.resize(model.dim_x);
    param.x_target << 1.5, 5.0, 0.0, 0.0, 0.0, 0.0;
    param.N = 10000;
    param.gamma_u = 10.0;
    Eigen::VectorXd sigma_u(model.dim_u);
    sigma_u << 1.5, 1.5, 1.5;
    param.sigma_u = sigma_u.asDiagonal();

    int maxiter = 200;
    
    for (int map = 299; map >= 0; --map) {
    // for (int map = 0; map < 300; ++map) {
        CollisionChecker collision_checker = CollisionChecker();
        collision_checker.loadMap("../BARN_dataset/txt_files/output_"+std::to_string(map)+".txt", 0.1);
        Solver solver(model);
        solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
        solver.U_0.row(2).array() += model.g;
        solver.init(param);
        solver.setCollisionChecker(&collision_checker);
        
        bool is_landed = false;
        bool is_failed = true;
        int i = 0;
        double total_elapsed = 0.0;
        double f_err = 0.0;
        for (i = 0; i < maxiter; ++i) {
            solver.solve();
            solver.move();
            
            // std::cout<<"1 solved in "<<solver.elapsed_1.count()<<std::endl;
            // std::cout<<"2 solved in "<<solver.elapsed_2.count()<<std::endl;
            // std::cout<<"3 solved in "<<solver.elapsed_3.count()<<std::endl;

            total_elapsed += solver.elapsed;

            if (collision_checker.getCollisionGrid(solver.x_init)) {
                // is_failed = true;
                break;
            }
            else {
                f_err = (solver.x_init.head(2) - param.x_target.head(2)).norm();
                if (solver.x_init(2) < 0) {
                    is_landed = true;
                    is_failed = false;
                    break;
                }
            }
            // std::cout<<solver.x_init.transpose()<<std::endl;
            // std::cout<<solver.U_0.col(0).transpose()<<std::endl;
            // solver.show();
        }
        std::cout<<map<<'\t'<<is_failed<<'\t'<<is_landed<<'\t'<<i<<'\t'<<total_elapsed<<'\t'<<f_err<<std::endl;
        // solver.showTraj();
    }
    return 0;
}
