# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

from typing import Any

import numpy as np
import numpy.typing as npt


def _check_valid_array(array_in: npt.NDArray[Any]) -> bool:
    if np.isnan(array_in).any() or np.isinf(array_in).any():
        msg = "k is NaN or Infinity"
        raise ValueError(msg)

    return True
