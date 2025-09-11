"""Routine to analyse q data."""

import logging

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import sisl
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit, minimize_scalar, root_scalar

logger = logging.getLogger(__name__)


MIN_LENGTH_FIT = 5
FIT_CUTOFF = 0.01


def _lambda_from_xi(xi: float, jdp: float) -> float:
    return np.sqrt(2 / (3 * np.sqrt(3) * xi * jdp))


def _correl_length_temp_dependence(
    temp: npt.NDArray[np.floating], xi_0: float, crit_temp: float
) -> npt.NDArray[np.floating]:
    return xi_0 / np.sqrt(1 - temp / crit_temp)


def _london_depth_temp_dependence(
    temp: npt.NDArray[np.floating], lambda_0: float, crit_temp: float
) -> npt.NDArray[np.floating]:
    return lambda_0 / np.sqrt(1 - (temp / crit_temp))


def get_lengths_vs_temp(
    q_data: dict[str, pd.DataFrame],
    hamiltonian: sisl.Hamiltonian,
) -> tuple[pd.DataFrame, matplotlib.figure.Figure]:
    """Calculate SC lengths vs temperature.

    Args:
        q_data (dict[str, pd.DataFrame]): Dataframe with q data.
        hamiltonian (sisl.Hamiltonian): Hamiltonian

    Returns
    -------
        tuple[pd.DataFrame, matplotlib.figure.Figure]: Results. Figure with fits.
    """
    lengths_row_list = []

    for temperature, data in q_data.items():
        lengths_dict = {
            "T": float(temperature.split("_")[-1]),
        }
        results_fit = data[
            np.abs(data["current_abs"]) / np.max(np.abs(data["current_abs"])) > FIT_CUTOFF
        ]
        results_fit.reset_index(drop=True, inplace=True)

        if len(results_fit) > MIN_LENGTH_FIT:
            j_spl = CubicSpline(x=results_fit["q_fraction"], y=results_fit["current_abs"])
            res = minimize_scalar(
                lambda x, spline=j_spl: -spline(x),
                bounds=(0, results_fit["q_fraction"].tail(1).item()),
            )
            q_j_max = float(res.x)
            j_dp = float(j_spl(q_j_max))
            lengths_dict.update({"q_j_max": q_j_max, "j_dp": j_dp})
            for orbital in range(hamiltonian.no):
                delta_spl = CubicSpline(
                    x=results_fit["q_fraction"],
                    y=np.abs(results_fit[f"delta_{orbital}"])
                    / np.abs(data.at[0, f"delta_{orbital}"]),
                )
                try:
                    res = root_scalar(
                        lambda x, spline=delta_spl: spline(x) - 1 / np.sqrt(2),
                        bracket=(0, results_fit["q_fraction"].tail(1)),
                    )

                    xi = 1 / (np.sqrt(2) * res.root * np.linalg.norm(hamiltonian.geometry.rcell[0]))
                    lengths_dict.update(
                        {
                            f"Q_{orbital}": res.root,
                            f"delta_{orbital}": delta_spl(res.root)
                            * np.abs(data.at[0, f"delta_{orbital}"]),
                            f"xi_{orbital}": xi,
                        },
                    )
                    if j_dp is not None:
                        lambda_london = _lambda_from_xi(xi, j_dp)
                        lengths_dict.update({f"lambda_{orbital}": lambda_london})
                except ValueError:
                    logger.exception("Value error.")
        lengths_row_list.append(lengths_dict)

    lengths_vs_temp = pd.DataFrame(lengths_row_list).sort_values("T").reset_index(drop=True)

    gap_and_current_fig, gap_and_current_axs = plt.subplots(
        ncols=hamiltonian.no + 1,
        figsize=(7 * hamiltonian.no, 5),
    )

    for temperature, data in q_data.items():
        for orbital in range(hamiltonian.no):
            gap_ax = gap_and_current_axs[orbital]
            if f"delta_{orbital}" in data:
                gap_ax.plot(
                    data["q_fraction"],
                    data[f"delta_{orbital}"],
                    "x--",
                    label=f"{float(temperature.split('_')[-1]):.2f}",
                )
            if f"Q_{orbital}" in lengths_vs_temp:
                gap_ax.plot(lengths_vs_temp[f"Q_{orbital}"], lengths_vs_temp[f"delta_{orbital}"], "o--")
            gap_ax.legend()

        current_ax = gap_and_current_axs[hamiltonian.no]
        current_ax.plot(
            data["q_fraction"],
            data["current_abs"],
            "x--",
            label=f"{float(temperature.split('_')[-1]):.2f}",
        )
        current_ax.legend()

    if "q_j_max" in lengths_vs_temp:
        current_ax = gap_and_current_axs[hamiltonian.no]
        current_ax.plot(lengths_vs_temp["q_j_max"], lengths_vs_temp["j_dp"], "o--")

    return lengths_vs_temp, gap_and_current_fig


def get_zero_temperature_values(
    hamiltonian: sisl.Hamiltonian, lengths_vs_temp: pd.DataFrame
) -> tuple[pd.DataFrame, matplotlib.figure.Figure]:
    """Get zero temperature values for the SC length scales.

    Args:
        hamiltonian (sisl.Hamiltonian): Hamiltonian.
        lengths_vs_temp (pd.DataFrame): SC lengths against temperature.

    Returns
    -------
        tuple[pd.DataFrame, matplotlib.figure.Figure]: Results, Figure with fits.
    """
    length_vs_temp_fig, length_vs_temp_axs = plt.subplots(
        nrows=2,
        ncols=hamiltonian.no,
        figsize=(7 * hamiltonian.no, 2 * 5),
    )
    zero_temp_length_row_list = []
    zero_temp_length_dict = {}
    for orbital in range(hamiltonian.no):
        xi_ax = length_vs_temp_axs[0, orbital]
        lambda_ax = length_vs_temp_axs[1, orbital]

        if f"xi_{orbital}" in lengths_vs_temp:
            xi_ax.plot(lengths_vs_temp["T"], lengths_vs_temp[f"xi_{orbital}"], "x--")

            xi_fit = lengths_vs_temp[["T", f"xi_{orbital}"]].dropna().reset_index(drop=True)
            xi_fit.reset_index(drop=True, inplace=True)

            if len(xi_fit) > MIN_LENGTH_FIT:
                p0, p0cov = curve_fit(
                    _correl_length_temp_dependence,
                    xi_fit["T"],
                    xi_fit[f"xi_{orbital}"],
                    bounds=([0.0, 0.0], [np.inf, np.inf]),
                    p0=[2.0, 2.0],
                )
                xi_0 = p0[0]
                crit_temp_xi = p0[1]
                temp_points_interpolate = np.linspace(
                    xi_fit.at[0, "T"],
                    xi_fit.at[len(xi_fit) - 1, "T"],
                    num=500,
                )
                xi_ax.plot(
                    temp_points_interpolate,
                    _correl_length_temp_dependence(temp_points_interpolate, xi_0, crit_temp_xi),
                )
                xi_ax.axvline(x=crit_temp_xi, ls="--")
                xi_ax.axhline(y=xi_0, ls="--")
                xi_ax.set_ylim(bottom=0)
                zero_temp_length_dict.update(
                    {f"xi0_{orbital}": xi_0, f"T_C_{orbital}_xi": crit_temp_xi},
                )
        if f"lambda_{orbital}" in lengths_vs_temp:
            lambda_ax.plot(lengths_vs_temp["T"], lengths_vs_temp[f"lambda_{orbital}"], "x--")
            lambda_fit = lengths_vs_temp[["T", f"lambda_{orbital}"]].dropna().reset_index(drop=True)

            if len(lambda_fit) > MIN_LENGTH_FIT:
                p0, p0cov = curve_fit(
                    _london_depth_temp_dependence,
                    lambda_fit["T"],
                    lambda_fit[f"lambda_{orbital}"],
                    bounds=([0.0, 0.0], [np.inf, np.inf]),
                    p0=[2.0, 2.0],
                )

                lambda_0 = p0[0]
                crit_temp_lambda = p0[1]
                temp_points_interpolate = np.linspace(
                    lambda_fit.at[0, "T"],
                    lambda_fit.at[len(lambda_fit) - 1, "T"],
                    num=500,
                )
                lambda_ax.plot(
                    temp_points_interpolate,
                    _london_depth_temp_dependence(
                        temp_points_interpolate, lambda_0, crit_temp_lambda
                    ),
                )
                lambda_ax.axvline(x=crit_temp_lambda, ls="--")
                lambda_ax.axhline(y=lambda_0, ls="--")
                lambda_ax.set_ylim(bottom=0)
                zero_temp_length_dict.update(
                    {
                        f"lambda0_{orbital}": lambda_0,
                        f"T_C_lambda0_{orbital}": crit_temp_lambda,
                    },
                )

    zero_temp_length_row_list.append(zero_temp_length_dict)
    zero_temp_lengths = pd.DataFrame(zero_temp_length_row_list)

    return zero_temp_lengths, length_vs_temp_fig
