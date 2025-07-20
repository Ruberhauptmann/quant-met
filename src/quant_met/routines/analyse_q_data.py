import sisl
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
import numpy as np
import logging


logger = logging.getLogger(__name__)


def analyse_q_data(
    q_data: dict[str, pd.DataFrame],
    hamiltonian: sisl.Hamiltonian,
):
    lengths_row_list = []
    q_extraction_list = []

    for temperature, data in q_data.items():
        q_extraction_dict = {"T": temperature}
        lengths_dict = {
            "T": temperature,
        }


        for orbital in range(hamiltonian.no):
            results_fit = data[
                np.abs(data[f"delta_{orbital}"]) / np.max(np.abs(data[f"delta_{orbital}"])) > 0.01
            ]
            results_fit.reset_index(drop=True, inplace=True)

            if len(results_fit) > 5:
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

                    q_extraction_dict.update(
                        {
                            f"Q_{orbital}": res.root,
                            f"Delta_{orbital}": delta_spl(res.root)
                            * np.abs(data.at[0, f"delta_{orbital}"]),
                        }
                    )

                    xi = 1 / (
                        np.sqrt(2)
                        * res.root
                        * np.linalg.norm(hamiltonian.geometry.rcell[0])
                    )
                    lengths_dict.update(
                        {
                            f"xi_{orbital}_a": xi,
                        }
                    )
                except ValueError:
                    logger.error("Value error.")
                    print("Value error")

        lengths_row_list.append(lengths_dict)
        q_extraction_list.append(q_extraction_dict)

    lengths_vs_temp = pd.DataFrame(lengths_row_list).sort_values("T").reset_index(drop=True)
    q_extraction_list = pd.DataFrame(q_extraction_list).sort_values("T").reset_index(drop=True)
    print(lengths_vs_temp, flush=True)
    print(q_extraction_list, flush=True)

    return None
