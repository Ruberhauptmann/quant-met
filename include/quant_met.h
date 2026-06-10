#pragma once
#include <Eigen/Dense>

// Toy function: Scales a matrix using OpenMP parallel loops
Eigen::MatrixXcd scale_matrix_parallel(const Eigen::MatrixXcd& input, double factor);
