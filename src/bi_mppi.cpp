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
    sigma_u << 0.0, 0.3;
    param.sigma_u = sigma_u.asDiagonal();
    param.deviation_mu = 1.0;
    param.cost_mu = 1.0;
    param.minpts = 5;
    param.epsilon = 0.01;
    param.psi = 0.8;

    int maxiter = 500;

    // for (int map = 299; map >= 0 ; --map) {
    // for (int map = 276; map >= 0; --map) {
    for (int s = 2; s < 3; ++s) {
    // for (int s = 1; s < 2; ++s) {
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
            double total_elapsed_1 = 0.0;
            double total_elapsed_2 = 0.0;
            double total_elapsed_3 = 0.0;
            for (i = 0; i < maxiter; ++i) {
                Solver solver(model);
                solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
                solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
                solver.init(param);
                solver.setCollisionChecker(&collision_checker);

                solver.x_init = x_prev;
                solver.x_target(1) = 5.0 + (0.05 * float(i));

                solver.solve();
                solver.move();
                x_prev = solver.x_init;
                total_elapsed += solver.elapsed;
                total_elapsed_1 += solver.elapsed_1.count();
                total_elapsed_2 += solver.elapsed_2.count();
                total_elapsed_3 += solver.elapsed_3.count();
                if (collision_checker.getCollisionGrid(solver.x_init)) {
                    break;
                }
                if (solver.x_init(1) > 5.0) {
                    is_success = true;
                    break;
                }
                // std::cout<<solver.x_init.transpose()<<std::endl;
                // solver.show();
            }
            std::cout<<s<<'\t'<<map<<'\t'<<i<<'\t'<<is_success<<'\t'<<total_elapsed<<std::endl;
            // std::cout<<total_elapsed_1<<'\t'<<total_elapsed_2<<'\t'<<total_elapsed_3<<std::endl;
        }
    }    

    return 0;
}
