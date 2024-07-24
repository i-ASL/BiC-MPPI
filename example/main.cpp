#include "invpend.h"
#include "cartpole.h"
#include "mppi.h"
#include "collision_checker.h"

#include "visualize.h"

int main() {
    // auto model = InvPend();
    auto model = CartPole();

    clock_t start = clock();

    MPPI mppi(model);
    MPPIParam mppi_param;
    mppi_param.Nu = 100;
    mppi_param.gamma_u = 1.0;
    mppi_param.sigma_u = 1.0;
    mppi.init(mppi_param);
    CollisionChecker *collision_checker;
    mppi.setCollisionChecker(collision_checker);
    mppi.solve();

    clock_t finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "\nIn Total : " << duration << " Seconds" << std::endl;

    // Parse Result
    Eigen::MatrixXd X_init = mppi.getInitX();
    Eigen::MatrixXd U_init = mppi.getInitU();
    Eigen::MatrixXd X_result = mppi.getResX();
    Eigen::MatrixXd U_result = mppi.getResU();
    std::vector<double> all_cost = mppi.getAllCost();

    // Visualize
    visualize(X_init, U_init, X_result, U_result, all_cost);

    return 0;
}