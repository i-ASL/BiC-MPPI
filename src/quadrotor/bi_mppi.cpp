#include <quadrotor.h>
#include <bi_mppi.h>

#include <iostream>
#include <Eigen/Dense>
#include <chrono>

int main() {
    auto model = Quadrotor();

    using Solver = BiMPPI;
    using SolverParam = BiMPPIParam;
    
    SolverParam param;
    param.dt = 0.1;
    param.Tf = 50;
    param.Tb = 50;
    param.x_init.resize(model.dim_x);
    param.x_init << 1.5, 0.0, 5.0, 0.0, 0.0, 0.0;
    param.x_target.resize(model.dim_x);
    param.x_target << 1.5, 5.0, 0.0, 0.0, 0.0, 0.0;
    param.Nf = 5000;
    param.Nb = 5000;
    param.Nr = 5000;
    param.gamma_u = 10.0;
    Eigen::VectorXd sigma_u(model.dim_u);
    sigma_u << 1.5, 1.5, 1.5;
    param.sigma_u = sigma_u.asDiagonal();
    param.deviation_mu = 1.0;
    param.cost_mu = 1.0;
    param.minpts = 5;
    param.epsilon = 0.01;
    param.psi = 0.6;

    int maxiter = 200;

    for (int map = 299; map >= 0; --map) {
        CollisionChecker collision_checker = CollisionChecker();
        collision_checker.loadMap("../BARN_dataset/txt_files/output_"+std::to_string(map)+".txt", 0.1);
        Solver solver(model);
        solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
        solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
        solver.U_f0.row(2).array() += model.g;
        solver.U_b0.row(2).array() += model.g;
        solver.init(param);
        solver.setCollisionChecker(&collision_checker);
        solver.dummy_u(2) += model.g;
        
        bool is_landed = false;
        bool is_failed = false;
        int i = 0;
        double total_elapsed  = 0.0;
        double f_err = 0.0;
        for (i = 0; i < maxiter; ++i) {
            solver.solve();
            solver.move();
            total_elapsed += solver.elapsed;
            
            // std::cout<<"1 solved in "<<solver.elapsed_1.count()<<std::endl;
            // std::cout<<"2 solved in "<<solver.elapsed_2.count()<<std::endl;
            // std::cout<<"3 solved in "<<solver.elapsed_3.count()<<std::endl;
            
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
            // solver.show();
        }
        std::cout<<map<<'\t'<<is_failed<<'\t'<<is_landed<<'\t'<<i<<'\t'<<total_elapsed<<'\t'<<f_err<<std::endl;
    }

    return 0;
}
