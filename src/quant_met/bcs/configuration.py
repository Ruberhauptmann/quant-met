from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class Configuration:
    t_gr: float
    t_x: float
    a: float
    V: float
    mu: float
    U_Gr: float
    U_X: float

    @property
    def U(self) -> list[float]:
        return [self.U_Gr, self.U_Gr, self.U_X]


class DeltaVector:
    def __init__(self, k_points: npt.NDArray, initial: float | None):
        self.k_points = k_points
        self.data = pd.DataFrame(
            columns=["kx", "ky", "delta_1", "delta_2", "delta_3"],
            index=range(len(k_points)),
            dtype=np.float64,
        )
        self.data.loc[:, "kx"] = self.k_points[:, 0]
        self.data.loc[:, "ky"] = self.k_points[:, 1]
        if initial is not None:
            self.data.loc[:, "delta_1"] = initial
            self.data.loc[:, "delta_2"] = initial
            self.data.loc[:, "delta_3"] = initial

    def __repr__(self):
        return self.data.to_string(index=False)

    def update_from_1d_vector(self, delta: npt.NDArray):
        for n in range(3):
            offset = int(n * len(delta) / 3)
            self.data.loc[:, f"delta_{n+1}"] = delta[
                offset : offset + len(self.k_points)
            ]

    def save(self, path):
        pass

    @property
    def as_1d_vector(self) -> npt.NDArray:
        return np.concatenate(
            [
                np.array(self.data.loc[:, "delta_1"].values),
                np.array(self.data.loc[:, "delta_2"].values),
                np.array(self.data.loc[:, "delta_3"].values),
            ]
        )
