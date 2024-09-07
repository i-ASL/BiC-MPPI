#include <wmrobot_map.h>
#include <bi_mppi.h>

#include <iostream>
#include <Eigen/Dense>
#include <chrono>

int main() {
    auto model = WMRobotMap();

    using Solver = BiMPPI;
    using SolverParam = BiMPPIParam;
    
    SolverParam param;
    param.dt = 0.1;
    param.Tf = 50;
    param.Tb = 50;
    param.x_init.resize(model.dim_x);
    param.x_init << 2.5, 0.0, M_PI_2;
    param.x_target.resize(model.dim_x);
    param.x_target << 1.5, 5.0, M_PI_2;
    param.Nf = 3000;
    param.Nb = 3000;
    param.Nr = 3000;
    param.gamma_u = 10.0;
    Eigen::VectorXd sigma_u(model.dim_u);
    sigma_u << 0.25, 0.25;
    param.sigma_u = sigma_u.asDiagonal();
    param.deviation_mu = 1.0;
    param.cost_mu = 1.0;
    param.minpts = 5;
    param.epsilon = 0.01;
    param.psi = 0.6;

    int maxiter = 200;
    
    for (int s = 0; s < 3; ++s) {
        switch (s)
        {
        case 0:
            param.x_init(0) = 0.5;
            break;
        case 1:
            param.x_init(0) = 1.5;
            break;
        case 2:
            param.x_init(0) = 2.5;
            break;
        default:
            break;
        }
        // for (int map = 0; map < 300; ++map) {
        for (int map = 299; map >= 0; --map) {
            CollisionChecker collision_checker = CollisionChecker();
            collision_checker.loadMap("../BARN_dataset/txt_files/output_"+std::to_string(map)+".txt", 0.1);
            Solver solver(model);
            solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
            solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
            solver.init(param);
            solver.setCollisionChecker(&collision_checker);
            
            bool is_success = false;
            bool is_collision = false;
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
                    is_collision = true;
                    break;
                }
                else {
                    f_err = (solver.x_init - param.x_target).norm();
                    if (f_err < 0.1) {
                        is_success = true;
                        break;
                    }
                }
            }
            std::cout<<s<<'\t'<<map<<'\t'<<is_success<<'\t'<<i<<'\t'<<total_elapsed<<std::endl;
        }
    }    

    return 0;
}
