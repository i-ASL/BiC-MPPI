#include <wmrobot_map.h>
#include <mppi.h>

#include <iostream>
#include <Eigen/Dense>
#include <chrono>

int main() {
    auto model = WMRobotMap();
    
    using Solver = MPPI;
    using SolverParam = MPPIParam;

    SolverParam param;
    param.dt = 0.1;
    param.T = 100;
    param.x_init.resize(model.dim_x);
    param.x_init << 2.5, 0.0, M_PI_2;
    param.x_target.resize(model.dim_x);
    param.x_target << 1.5, 5.0, M_PI_2;
    param.N = 6000;
    param.gamma_u = 10.0;
    Eigen::VectorXd sigma_u(model.dim_u);
    sigma_u << 0.0, 0.3;
    param.sigma_u = sigma_u.asDiagonal();

    int maxiter = 500;
    
    // for (int map = 299; map >= 0 ; --map) {
    // for (int map = 276; map >= 0; --map) {
    // for (int s = 1; s < 2; ++s) {
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
        for (int map = 299; map >= 0; --map) {
        // for (int map = 0; map < 300; ++map) {
            CollisionChecker collision_checker = CollisionChecker();
            collision_checker.loadMap("../BARN_dataset/txt_files/output_"+std::to_string(map)+".txt", 0.1);
            
            bool is_success = false;
            int i = 0;
            Eigen::VectorXd x_prev = param.x_init;
            double total_elapsed = 0.0;
            for (i = 0; i < maxiter; ++i) {
                Solver solver(model);
                solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
                solver.init(param);
                solver.setCollisionChecker(&collision_checker);

                solver.x_init = x_prev;
                solver.x_target(1) = 5.0 + (0.05 * float(i));

                solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
                solver.solve();
                solver.move();
                
                // std::cout<<"1 solved in "<<solver.elapsed_1.count()<<std::endl;
                // std::cout<<"2 solved in "<<solver.elapsed_2.count()<<std::endl;
                // std::cout<<"3 solved in "<<solver.elapsed_3.count()<<std::endl;

                x_prev = solver.x_init;
                total_elapsed += solver.elapsed;
                
                // std::cout<<solver.getX().transpose()<<std::endl;
                if (collision_checker.getCollisionGrid(solver.x_init)) {
                    break;
                }
                if (solver.x_init(1) > 5.0) {
                    is_success = true;
                    break;
                }
            }
            std::cout<<s<<'\t'<<map<<'\t'<<i<<'\t'<<is_success<<'\t'<<total_elapsed<<std::endl;
            // solver.showTraj();
        }
    }    

    return 0;
}
