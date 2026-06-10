#include "quant_met.h"
#include <omp.h>

Eigen::MatrixXcd scale_matrix_parallel(const Eigen::MatrixXcd& input, double factor) {
    Eigen::MatrixXcd result = input;

    // Simple OpenMP test to verify multi-threading works
#pragma omp parallel for collapse(2)
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            result(i, j) *= factor;
        }
    }

    return result;
}