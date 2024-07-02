# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

import numpy as np


def _check_valid_float(float_in: float, parameter_name: str) -> float:
    if np.isinf(float_in):
        raise ValueError(f"{parameter_name} must not be Infinity")
    elif np.isnan(float_in):
        raise ValueError(f"{parameter_name} must not be NaN")
    else:
        return float_in
