#include <pybind11/pybind11.h>
#include <Eigen/Core>

namespace py = pybind11;

Eigen::MatrixXcd scale_quantum_matrix(const Eigen::MatrixXcd& input_matrix, double scaling_factor) {
    // Standard validation guardrail
    if (input_matrix.rows() == 0 || input_matrix.cols() == 0) {
        return input_matrix;
    }

    // Direct element-wise operation (Eigen automatically optimizes this under the hood)
    Eigen::MatrixXcd result = input_matrix * scaling_factor;

    return result;
}
