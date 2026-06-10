# src/quant_met_core.pyi
import numpy as np
import typing

def scale_matrix_parallel(input: np.ndarray, factor: float) -> np.ndarray:
    """Scales a complex matrix concurrently using OpenMP."""