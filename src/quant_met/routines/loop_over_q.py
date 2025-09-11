"""Function to run search for critical temperature."""

import logging
from functools import partial
from multiprocessing import Pool
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import sisl

from quant_met import bdg

from .self_consistency import self_consistency_loop

logger = logging.getLogger(__name__)

MAX_Q = 0.5


def is_gap_zero(result: dict[str, float], atol: float = 1e-8) -> bool:
    """Check if all delta values in result are (approximately) zero."""
    deltas = np.array([x for key, x in result.items() if key.startswith("delta")])
    return np.isclose(np.max(np.abs(deltas)), 0, atol=atol)


def adjust_q_upper_bound(
    gap_for_q_partial: partial[dict[str, Any] | None], initial_q: float = 0.5
) -> float:
    """
    Adjust q_upper_bound until gap is non-zero or exceeds upper limit.

    Returns the adjusted q_upper_bound.
    """
    q_upper_bound = initial_q

    while True:
        result_tmp: dict[str, float] | None = gap_for_q_partial(q_upper_bound)

        if result_tmp is None or is_gap_zero(result_tmp):
            q_upper_bound /= 2
        else:
            break

    while True:
        result_tmp = gap_for_q_partial(q_upper_bound)

        if result_tmp is None or is_gap_zero(result_tmp):
            break
        else:
            q_upper_bound *= 1.1
            if q_upper_bound > MAX_Q:
                q_upper_bound = MAX_Q
                break

    return q_upper_bound


def _gap_for_q(  # noqa: PLR0913
    q_fraction: float,
    hamiltonian: sisl.Hamiltonian,
    kgrid: sisl.MonkhorstPack,
    hubbard_int_orbital_basis: npt.NDArray[np.float64],
    epsilon: float,
    temp: float,
    max_iter: int = 1000,
) -> dict[str, Any] | None:  # pragma: no cover
    beta = np.inf if temp == 0 else 1 / temp
    q = np.array([q_fraction, 0, 0])
    data_dict: dict[str, Any] = {
        "q_fraction": q_fraction,
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
        bdg_energies, bdg_wavefunctions = bdg.diagonalize_bdg(
            hamiltonian=hamiltonian,
            kgrid=kgrid,
            delta_orbital_basis=gap,
            q=q,
        )
        current = bdg.calculate_current_density(
            hamiltonian=hamiltonian,
            k=kgrid,
            bdg_energies=bdg_energies,
            bdg_wavefunctions=bdg_wavefunctions,
            beta=beta,
        )
        data_dict.update({f"delta_{orbital}": gap[orbital] for orbital in range(len(gap))})
        data_dict.update(
            {
                "current_x": current[0],
                "current_y": current[1],
                "current_abs": np.linalg.norm(current),
            },
        )
        return data_dict


def loop_over_q(  # noqa: PLR0913
    hamiltonian: sisl.Hamiltonian,
    kgrid: sisl.MonkhorstPack,
    hubbard_int_orbital_basis: npt.NDArray[np.float64],
    epsilon: float,
    max_iter: int,
    n_q_points: int,
    crit_temps: npt.NDArray[np.float64],
) -> dict[str, pd.DataFrame]:  # pragma: no cover
    """Loop over q."""
    logger.info("Start search for upper bound for q.")

    crit_temp = np.max(crit_temps)
    temp_list = [crit_temp * x for x in [0.65, 0.7, 0.75, 0.8, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95]]

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
        q_upper_bound = adjust_q_upper_bound(gap_for_q_partial, initial_q=0.5)
        logger.info("q upper bound: %s", q_upper_bound)

        q_list = np.linspace(
            0,
            q_upper_bound,
            num=n_q_points,
        )

        with Pool() as p:
            delta_vs_q_list = [x for x in p.map(gap_for_q_partial, q_list) if x is not None]  # type: ignore[arg-type]

        delta_vs_q_tmp = (
            pd.DataFrame(delta_vs_q_list).sort_values(by=["q_fraction"]).reset_index(drop=True)
        )
        delta_vs_q[f"{temp}"] = delta_vs_q_tmp

    return delta_vs_q
