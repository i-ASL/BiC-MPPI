#include <wmrobot_map.h>
#include <bi_mppi.h>

#include <iostream>
#include <Eigen/Dense>
#include <chrono>

int main() {
    auto model = WMRobotMap();

    BiMPPIParam bi_mppi_param;
    bi_mppi_param.dt = 0.2;
    bi_mppi_param.Tf = 50;
    bi_mppi_param.Tb = 50;
    bi_mppi_param.x_init.resize(model.dim_x);
    bi_mppi_param.x_init << 1.5, 0.0, M_PI_2;
    bi_mppi_param.x_target.resize(model.dim_x);
    bi_mppi_param.x_target << 1.5, 5.0, M_PI_2;
    bi_mppi_param.Nf = 3000;
    bi_mppi_param.Nb = 3000;
    bi_mppi_param.Nr = 3000;
    bi_mppi_param.gamma_u = 10.0;
    Eigen::VectorXd sigma_u(model.dim_u);
    sigma_u << 0.0, 0.5;
    bi_mppi_param.sigma_u = sigma_u.asDiagonal();
    bi_mppi_param.deviation_mu = 1.0;
    bi_mppi_param.cost_mu = 1.0;
    bi_mppi_param.minpts = 5;
    bi_mppi_param.epsilon = 0.01;
    bi_mppi_param.psi = 0.7;

    // for (int i = 299; i >= 0 ; --i) {
    for (int i = 278; i >= 0 ; i) {
    // for (int i = 0; i < 300; ++i) {
        CollisionChecker collision_checker = CollisionChecker();
        collision_checker.loadMap("../BARN_dataset/txt_files/output_"+std::to_string(i)+".txt", 0.1);
        BiMPPI bi_mppi(model);
        bi_mppi.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, bi_mppi_param.Tf);
        bi_mppi.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, bi_mppi_param.Tb);
        bi_mppi.init(bi_mppi_param);
        bi_mppi.setCollisionChecker(&collision_checker);
        
        while (true) {
            bi_mppi.solve();
            std::cout<<"1 solved in "<<bi_mppi.elapsed_1.count()<<std::endl;
            std::cout<<"2 solved in "<<bi_mppi.elapsed_2.count()<<std::endl;
            std::cout<<"3 solved in "<<bi_mppi.elapsed_3.count()<<std::endl;
            
            // std::cout<<bi_mppi.getX().transpose()<<std::endl;
            if (bi_mppi.getX()(1) > 5.0) {break;}
        }
        bi_mppi.showTraj();
    }

    return 0;
}
