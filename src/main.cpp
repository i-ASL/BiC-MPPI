#include "invpend.h"
#include "cartpole.h"
#include "mppi.h"

#include "visualize.h"

int main() {
    auto model = InvPend();
    // auto model = CartPole();

    clock_t start = clock();

    MPPI mppi(model);
    int Nu = 100;
    double lambda = 1.0;
    double sigma_u = 1.0;
    mppi.init(Nu, lambda, sigma_u);
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