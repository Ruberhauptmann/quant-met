import logging

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sisl
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit, minimize_scalar, root_scalar

logger = logging.getLogger(__name__)


def lambda_from_xi(xi, jdp):
    return np.sqrt(2 / (3 * np.sqrt(3) * xi * jdp))


def correl_length_T_dependence(T, xi_0, T_C):
    return xi_0 / np.sqrt(1 - T / T_C)


def london_depth_T_dependence(T, lambda_0, T_C):
    return lambda_0 / np.sqrt(1 - (T / T_C))


def get_lengths_vs_temp(
    q_data: dict[str, pd.DataFrame],
    hamiltonian: sisl.Hamiltonian,
) -> tuple[pd.DataFrame, matplotlib.figure.Figure]:
    lengths_row_list = []

    for temperature, data in q_data.items():
        lengths_dict = {
            "T": float(temperature.split("_")[-1]),
        }
        results_fit = data[np.abs(data["current_abs"]) / np.max(np.abs(data["current_abs"])) > 0.01]
        results_fit.reset_index(drop=True, inplace=True)

        if len(results_fit) > 5:
            j_spl = CubicSpline(x=results_fit["q_fraction"], y=results_fit["current_abs"])
            res = minimize_scalar(
                lambda x: -j_spl(x),
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
                        lambda x: delta_spl(x) - 1 / np.sqrt(2),
                        bracket=(0, results_fit["q_fraction"].tail(1)),
                    )

                    xi = 1 / (np.sqrt(2) * res.root * np.linalg.norm(hamiltonian.geometry.rcell[0]))
                    lengths_dict.update(
                        {
                            f"Q_{orbital}": res.root,
                            f"delta_{orbital}": delta_spl(res.root)
                            * np.abs(data.at[0, f"delta_{orbital}"]),
                            f"xi_{orbital}": xi,
                        }
                    )
                    if j_dp is not None:
                        lambda_L = lambda_from_xi(xi, j_dp)
                        lengths_dict.update({f"lambda_{orbital}": lambda_L})
                except ValueError:
                    logger.error("Value error.")
                    print("Value error")
        lengths_row_list.append(lengths_dict)

    lengths_vs_temp = pd.DataFrame(lengths_row_list).sort_values("T").reset_index(drop=True)

    gap_and_current_fig, gap_and_current_axs = plt.subplots(
        ncols=hamiltonian.no + 1, figsize=(7 * hamiltonian.no, 5)
    )

    for temperature, data in q_data.items():
        for orbital in range(hamiltonian.no):
            gap_ax = gap_and_current_axs[orbital]
            gap_ax.plot(
                data["q_fraction"],
                data[f"delta_{orbital}"],
                "x--",
                label=f"{float(temperature.split('_')[-1]):.2f}",
            )
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

    for orbital in range(hamiltonian.no):
        current_ax = gap_and_current_axs[hamiltonian.no]
        current_ax.plot(lengths_vs_temp["q_j_max"], lengths_vs_temp["j_dp"], "o--")

    return lengths_vs_temp, gap_and_current_fig


def get_zero_temperature_values(hamiltonian: sisl.Hamiltonian, lengths_vs_temp: pd.DataFrame):
    length_vs_temp_fig, length_vs_temp_axs = plt.subplots(
        nrows=2, ncols=hamiltonian.no, figsize=(7 * hamiltonian.no, 2 * 5)
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

            if len(xi_fit) > 5:
                p0, p0cov = curve_fit(
                    correl_length_T_dependence,
                    xi_fit["T"],
                    xi_fit[f"xi_{orbital}"],
                    bounds=([0.0, 0.0], [np.inf, np.inf]),
                    p0=[2.0, 2.0],
                )
                if np.sqrt(np.diag(p0cov))[0] < 0.1:
                    xi_0 = p0[0]
                    T_C_xi = p0[1]
                    T_interpolate = np.linspace(
                        xi_fit.at[0, "T"], xi_fit.at[len(xi_fit) - 1, "T"], num=500
                    )
                    xi_ax.plot(
                        T_interpolate, correl_length_T_dependence(T_interpolate, xi_0, T_C_xi)
                    )
                    xi_ax.axvline(x=T_C_xi, ls="--")
                    xi_ax.axhline(y=xi_0, ls="--")
                    xi_ax.set_ylim(bottom=0)
                    zero_temp_length_dict.update(
                        {f"xi0_{orbital}": xi_0, f"T_C_{orbital}_xi": T_C_xi}
                    )
        if f"lambda_{orbital}" in lengths_vs_temp:
            lambda_ax.plot(lengths_vs_temp["T"], lengths_vs_temp[f"lambda_{orbital}"], "x--")
            lambda_fit = lengths_vs_temp[["T", f"lambda_{orbital}"]].dropna().reset_index(drop=True)

            if len(lambda_fit) > 5:
                p0, p0cov = curve_fit(
                    london_depth_T_dependence,
                    lambda_fit["T"],
                    lambda_fit[f"lambda_{orbital}"],
                    bounds=([0.0, 0.0], [np.inf, np.inf]),
                    p0=[2.0, 2.0],
                )
                if np.sqrt(np.diag(p0cov))[0] < 0.1:
                    lambda_0 = p0[0]
                    T_C_lambda = p0[1]
                    T_interpolate = np.linspace(
                        lambda_fit.at[0, "T"], lambda_fit.at[len(lambda_fit) - 1, "T"], num=500
                    )
                    lambda_ax.plot(
                        T_interpolate,
                        correl_length_T_dependence(T_interpolate, lambda_0, T_C_lambda),
                    )
                    lambda_ax.axvline(x=T_C_lambda, ls="--")
                    lambda_ax.axhline(y=lambda_0, ls="--")
                    lambda_ax.set_ylim(bottom=0)
                    zero_temp_length_dict.update(
                        {f"lambda0_{orbital}": lambda_0, f"T_C_lambda0_{orbital}": T_C_lambda}
                    )

    zero_temp_length_row_list.append(zero_temp_length_dict)
    zero_temp_lengths = pd.DataFrame(zero_temp_length_row_list)

    return zero_temp_lengths, length_vs_temp_fig
