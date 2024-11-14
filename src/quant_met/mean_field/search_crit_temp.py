# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Function to run search for critical temperature."""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

from quant_met.mean_field import self_consistency_loop
from quant_met.mean_field.hamiltonians import BaseHamiltonian
from quant_met.parameters import GenericParameters

logger = logging.getLogger(__name__)


def search_crit_temp(
    h: BaseHamiltonian[GenericParameters],
    k_space_grid: npt.NDArray[np.float64],
    epsilon: float,
    max_iter: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Search for critical temperature."""
    temp = 1 / h.beta if not np.isinf(h.beta) else 1 / (0.1 * h.delta_orbital_basis[0])

    delta_vs_temp_list = []
    critical_temperatures_list = []

    h.beta = np.inf
    solved_h = self_consistency_loop(h, k_space_grid, epsilon, max_iter)

    data_dict = {
        "T": 0,
    }
    zero_temperature_gap = solved_h.delta_orbital_basis
    data_dict.update(
        {
            f"delta_{orbital}": zero_temperature_gap[orbital]
            for orbital in range(len(zero_temperature_gap))
        }
    )
    delta_vs_temp_list.append(data_dict)

    logger.info("Starting temperature %s", temp)

    zero_gap_temp = nonzero_gap_temp = temp

    found_zero_gap = False
    found_nonzero_gap = False
    while (found_zero_gap and found_nonzero_gap) is False:
        logger.info("New temp: %s, ", temp)
        h.beta = 1 / temp
        solved_h = self_consistency_loop(h, k_space_grid, epsilon, max_iter)

        data_dict = {
            "T": temp,
        }
        data_dict.update(
            {
                f"delta_{orbital}": h.delta_orbital_basis[orbital]
                for orbital in range(h.number_of_bands)
            }
        )
        delta_vs_temp_list.append(data_dict)

        if np.allclose(solved_h.delta_orbital_basis, 0):
            zero_gap_temp = temp
            temp = 0.5 * temp
            logger.info("Found temperature with zero gap.")
            found_zero_gap = True
        elif np.allclose(solved_h.delta_orbital_basis, zero_temperature_gap):
            nonzero_gap_temp = temp
            temp = 2 * temp
            logger.info("Found temperature with nonzero gap.")
            found_nonzero_gap = True
        else:
            temp = 0.5 * temp

    logger.info("Temperature bounds: %s to %s", nonzero_gap_temp, zero_gap_temp)

    temperature_list = np.concatenate(
        [
            np.linspace(0.5 * nonzero_gap_temp, nonzero_gap_temp, num=3, endpoint=False),
            np.linspace(nonzero_gap_temp, zero_gap_temp, num=30, endpoint=False),
            np.linspace(zero_gap_temp, 1.5 * zero_gap_temp, num=3, endpoint=True),
        ]
    )
    print(temperature_list)

    delta_vs_temp = pd.DataFrame(delta_vs_temp_list).sort_values(by=["T"]).reset_index(drop=True)
    print(delta_vs_temp)
    critical_temperatures = pd.DataFrame(critical_temperatures_list)

    return delta_vs_temp, critical_temperatures
