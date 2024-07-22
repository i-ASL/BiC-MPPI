#pragma once

#include <eigen3/Eigen/Dense>

class CollisionChecker {
public:
    CollisionChecker();
    ~CollisionChecker();
    double getCost(const Eigen::VectorXd &x);
};

CollisionChecker::CollisionChecker() {
}

CollisionChecker::~CollisionChecker() {
}

double CollisionChecker::getCost(const Eigen::VectorXd &x) {
    return 0.0;
}