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
import sisl

from .self_consistency import self_consistency_loop

logger = logging.getLogger(__name__)


def _gap_for_q(
    q_fraction: float,
    hamiltonian: sisl.Hamiltonian,
    kgrid: sisl.MonkhorstPack,
    hubbard_int_orbital_basis: npt.NDArray[np.float64],
    epsilon: float,
    temp: float,
    max_iter: int = 1000,
) -> dict[str, Any] | None:  # pragma: no cover
    beta = np.inf if temp == 0 else 1 / temp
    q = q_fraction * hamiltonian.geometry.rcell[0]
    data_dict: dict[str, Any] = {
        "q": q_fraction,
    }
    try:
        gap = self_consistency_loop(
            hamiltonian=hamiltonian,
            kgrid=kgrid,
            beta=beta,
            hubbard_int_orbital_basis=hubbard_int_orbital_basis,
            epsilon=epsilon,
            max_iter=max_iter,
            q=q,
        )
    except RuntimeError:
        logger.exception("Did not converge.")
        return None
    else:
        data_dict.update({f"delta_{orbital}": gap[orbital] for orbital in range(len(gap))})
        return data_dict


def loop_over_q(
    hamiltonian: sisl.Hamiltonian,
    kgrid: sisl.MonkhorstPack,
    hubbard_int_orbital_basis: npt.NDArray[np.float64],
    epsilon: float,
    max_iter: int,
    n_q_points: int,
    crit_temps: npt.NDArray[np.float64],
) -> tuple[dict[str, pd.DataFrame], matplotlib.figure.Figure]:  # pragma: no cover
    """Loop over q."""
    logger.info("Start search for upper bound for q.")

    crit_temp = np.max(crit_temps)
    # temp_list = [crit_temp * x for x in [0.65, 0.7, 0.75, 0.8, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95]]
    temp_list = [crit_temp * x for x in [0.70, 0.80, 0.90]]

    fig, axs = plt.subplots(ncols=hamiltonian.no, figsize=(7 * hamiltonian.no, 5))

    delta_vs_q = {}
    for temp in temp_list:
        gap_for_q_partial = partial(
            _gap_for_q,
            hamiltonian=hamiltonian,
            kgrid=kgrid,
            hubbard_int_orbital_basis=hubbard_int_orbital_basis,
            epsilon=epsilon,
            max_iter=max_iter,
            temp=temp,
        )

        q_upper_bound = 0.5
        while True:
            result_tmp = gap_for_q_partial(q_upper_bound)
            if result_tmp is None:
                q_upper_bound = q_upper_bound / 2
            else:
                result_tmp = np.array(
                    [x for key, x in result_tmp.items() if key.startswith("delta")]
                )
                if np.isclose(np.max(np.abs(result_tmp)), 0, atol=1e-8):
                    q_upper_bound = q_upper_bound / 2
                else:
                    while True:
                        result_tmp = gap_for_q_partial(q_upper_bound)
                        if result_tmp is None:
                            q_upper_bound = q_upper_bound + 0.1 * q_upper_bound
                            if q_upper_bound > 0.5:
                                break
                        else:
                            result_tmp = np.array(
                                [x for key, x in result_tmp.items() if key.startswith("delta")]
                            )
                            if not np.isclose(np.max(np.abs(result_tmp)), 0, atol=1e-8):
                                q_upper_bound = q_upper_bound + 0.1 * q_upper_bound
                                if q_upper_bound > 0.5:
                                    break
                            else:
                                break
                break
        q_upper_bound = min(q_upper_bound, 0.5)
        logger.info("q upper bound: %s", q_upper_bound)

        q_list = np.linspace(
            0,
            q_upper_bound,
            num=n_q_points,
        )

        with Pool() as p:
            delta_vs_q_list = [x for x in p.map(gap_for_q_partial, q_list) if x is not None]  # type: ignore[arg-type]

        delta_vs_q_tmp = pd.DataFrame(delta_vs_q_list).sort_values(by=["q"]).reset_index(drop=True)
        delta_vs_q[f"{temp}"] = delta_vs_q_tmp

        for orbital in range(hamiltonian.no):
            ax = axs[orbital]
            ax.plot(
                delta_vs_q_tmp["q"], delta_vs_q_tmp[f"delta_{orbital}"], "x--", label=f"{temp:.2f}"
            )
            ax.legend()

    return delta_vs_q, fig
