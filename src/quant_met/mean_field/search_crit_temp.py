# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

"""Function to run search for critical temperature."""

import logging
from functools import partial
from multiprocessing import Pool
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from quant_met import plotting
from quant_met.parameters import GenericParameters

from .hamiltonians import BaseHamiltonian
from .self_consistency import self_consistency_loop

logger = logging.getLogger(__name__)


def _get_bounds(
    initial_temp: float,
    gap_for_temp_partial: partial[dict[str, Any] | None],
    zero_temperature_gap: npt.NDArray[np.complex64],
) -> tuple[list[dict[str, Any]], float, float]:  # pragma: no cover
    delta_vs_temp_list = []
    zero_gap_temp = nonzero_gap_temp = 0.0
    found_zero_gap = False
    found_nonzero_gap = False
    temp = initial_temp
    direction = "down"
    iterations = 0
    while (found_zero_gap and found_nonzero_gap) is False and iterations < 100:
        logger.info("Trying temperature: %s", temp)
        data_dict = gap_for_temp_partial(temp)
        if data_dict is not None:
            delta_vs_temp_list.append(data_dict)
            gap = np.array([data_dict[key] for key in data_dict if key.startswith("delta")])
            if np.allclose(gap, 0):
                logger.info("Found temperature with zero gap.")
                zero_gap_temp = temp
                found_zero_gap = True
                temp = 0.5 * temp
            elif np.allclose(gap, zero_temperature_gap, atol=np.min(np.abs(zero_temperature_gap))):
                logger.info("Found temperature with nonzero gap.")
                nonzero_gap_temp = temp
                found_nonzero_gap = True
                temp = 2 * temp
            elif direction == "down":
                logger.info(
                    "Gap is neither zero nor equal to the nonzero gap. Reducing temperature."
                )
                temp = 0.5 * temp
            else:
                logger.info(
                    "Gap is neither zero nor equal to the nonzero gap. Increasing temperature."
                )
                temp = 2 * temp
        elif direction == "down":
            logger.info("No data found for temperature %s. Reducing temperature.", temp)
            temp = 0.5 * temp
        else:
            logger.info("No data found for temperature %s. Increasing temperature.", temp)
            temp = 2 * temp

        if found_zero_gap and direction == "up" and not found_nonzero_gap:
            logger.info("Switching direction to decrease temperature.")
            temp = initial_temp / 2
            direction = "down"
        if found_nonzero_gap and direction == "down" and not found_zero_gap:
            logger.info("Switching direction to increase temperature.")
            temp = 2 * initial_temp
            direction = "up"

        iterations += 1
    return delta_vs_temp_list, zero_gap_temp, nonzero_gap_temp


def _fit_for_crit_temp(
    delta_vs_temp: pd.DataFrame, orbital: int
) -> tuple[pd.DataFrame | None, pd.DataFrame, float | None, float | None]:  # pragma: no cover
    filtered_results = delta_vs_temp.iloc[
        np.where(
            np.invert(
                np.logical_or(
                    np.isclose(
                        np.abs(delta_vs_temp[f"delta_{orbital}"]) ** 2,
                        0,
                        atol=1000 * (np.abs(delta_vs_temp[f"delta_{orbital}"]) ** 2).min(),
                        rtol=1e-3,
                    ),
                    np.isclose(
                        np.abs(delta_vs_temp[f"delta_{orbital}"]) ** 2,
                        (np.abs(delta_vs_temp[f"delta_{orbital}"]) ** 2).max(),
                        rtol=1e-3,
                    ),
                )
            )
        )
    ]

    err = []
    if len(filtered_results) <= 4:
        return None, filtered_results, None, None

    lengths = range(4, len(filtered_results))

    for length in lengths:
        range_results = filtered_results.iloc[-length:]
        linreg = stats.linregress(
            range_results["T"], np.abs(range_results[f"delta_{orbital}"]) ** 2
        )
        err.append(linreg.stderr)

    min_length = lengths[np.argmin(np.array(err))]
    range_results = filtered_results.iloc[-min_length:]
    linreg = stats.linregress(range_results["T"], np.abs(range_results[f"delta_{orbital}"]) ** 2)

    return range_results, filtered_results, linreg.intercept, linreg.slope


def _gap_for_temp(
    temp: float,
    h: BaseHamiltonian[GenericParameters],
    k_space_grid: npt.NDArray[np.float64],
    epsilon: float,
    max_iter: int = 1000,
) -> dict[str, Any] | None:  # pragma: no cover
    beta = np.inf if temp == 0 else 1 / temp
    h.beta = beta
    try:
        solved_h = self_consistency_loop(h, k_space_grid, epsilon, max_iter)
    except RuntimeError:
        logger.exception("Did not converge.")
        return None
    else:
        data_dict = {
            "T": temp,
        }
        zero_temperature_gap = solved_h.delta_orbital_basis
        data_dict.update(
            {
                f"delta_{orbital}": zero_temperature_gap[orbital]
                for orbital in range(len(zero_temperature_gap))
            }
        )
        return data_dict


def search_crit_temp(
    h: BaseHamiltonian[GenericParameters],
    k_space_grid: npt.NDArray[np.float64],
    epsilon: float,
    max_iter: int,
    n_temp_points: int,
) -> tuple[pd.DataFrame, list[float], matplotlib.figure.Figure]:  # pragma: no cover
    """Search for critical temperature."""
    logger.info("Start search for bounds for T_C")
    temp = 1 / h.beta if not np.isinf(h.beta) else 0.25 * h.hubbard_int_orbital_basis[0]

    delta_vs_temp_list = []
    critical_temp_list = []

    gap_for_temp_partial = partial(
        _gap_for_temp, h=h, k_space_grid=k_space_grid, epsilon=epsilon, max_iter=max_iter
    )

    logger.info("Calculating zero temperature gap")
    data_dict = gap_for_temp_partial(0)
    assert data_dict is not None

    zero_temperature_gap = np.array(
        [data_dict[key] for key in data_dict if key.startswith("delta")]
    )
    delta_vs_temp_list.append(data_dict)

    delta_vs_temp_list_tmp, zero_gap_temp, nonzero_gap_temp = _get_bounds(
        temp, gap_for_temp_partial, zero_temperature_gap
    )
    delta_vs_temp_list.extend(delta_vs_temp_list_tmp)
    logger.info("Temperature bounds: %s to %s", nonzero_gap_temp, zero_gap_temp)

    temperature_list = np.concatenate(
        [
            np.linspace(
                0.8 * nonzero_gap_temp,
                nonzero_gap_temp,
                num=int(0.05 * n_temp_points),
                endpoint=False,
            ),
            np.linspace(
                nonzero_gap_temp, zero_gap_temp, num=int(0.9 * n_temp_points), endpoint=False
            ),
            np.linspace(
                zero_gap_temp, 1.2 * zero_gap_temp, num=int(0.05 * n_temp_points), endpoint=True
            ),
        ]
    )

    with Pool() as p:
        delta_vs_temp_list.extend(p.map(gap_for_temp_partial, temperature_list))  # type: ignore[arg-type]
        delta_vs_temp_list = [x for x in delta_vs_temp_list if x is not None]

    delta_vs_temp = pd.DataFrame(delta_vs_temp_list).sort_values(by=["T"]).reset_index(drop=True)

    fit_fig, fit_axs = plt.subplots(
        nrows=1, ncols=h.number_of_bands, figsize=(h.number_of_bands * 6, 6)
    )

    for orbital in range(h.number_of_bands):
        fit_range, filtered_range, intercept, slope = _fit_for_crit_temp(delta_vs_temp, orbital)

        ax = fit_axs if h.number_of_bands == 1 else fit_axs[orbital]

        if fit_range is not None and intercept is not None and slope is not None:
            critical_temp = -intercept / slope
            critical_temp_list.append(critical_temp)

            ax.plot(
                filtered_range["T"],
                intercept + slope * filtered_range["T"],
                "r--",
                alpha=0.3,
            )
            ax.plot(
                fit_range["T"],
                intercept + slope * fit_range["T"],
                "r-",
            )
            ax.axvline(x=critical_temp, linestyle="--", color="gray")
        else:
            critical_temp = 0
            critical_temp_list.append(critical_temp)

        ax.plot(
            delta_vs_temp["T"],
            np.abs(delta_vs_temp[f"delta_{orbital}"]) ** 2,
            "--x",
            color=f"C{orbital}",
        )
        ax = plotting.format_plot(ax)
        ax.set_ylabel(r"$\vert\Delta\vert^2\ [t^2]$")

    return delta_vs_temp, critical_temp_list, fit_fig
