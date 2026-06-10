#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> // Vital for handling Eigen <-> NumPy conversion
#include "quant_met.h"

namespace py = pybind11;

PYBIND11_MODULE(quant_met_core, m) {
    m.doc() = "quant_met C++ computation engine";

    m.def("scale_matrix_parallel", &scale_matrix_parallel,
          "Scales a complex matrix concurrently using OpenMP",
          py::arg("input"), py::arg("factor"));
}