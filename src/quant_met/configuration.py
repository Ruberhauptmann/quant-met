import numpy as np
import numpy.typing as npt
import pandas as pd


class DeltaVector:
    def __init__(
        self,
        number_bands: int,
        hdf_file=None,
        k_points: npt.NDArray | None = None,
        initial: float | None = None,
    ):
        self.number_bands = number_bands
        if hdf_file is not None:
            self.data = pd.DataFrame(pd.read_hdf(hdf_file, key="table"))
            self.k_points = np.column_stack(
                (np.array(self.data.loc[:, "kx"]), np.array(self.data.loc[:, "ky"]))
            )
        else:
            self.k_points = k_points
            self.data = pd.DataFrame(
                # columns=["kx", "ky", "delta_1", "delta_2", "delta_3"],
                index=range(len(k_points)),
                dtype=np.float64,
            )
            self.data.loc[:, "kx"] = self.k_points[:, 0]
            self.data.loc[:, "ky"] = self.k_points[:, 1]
            if initial is not None:
                for i in range(number_bands):
                    self.data.loc[:, f"delta_{i}"] = initial

    def __repr__(self):
        return self.data.to_string(index=False)

    def update_from_1d_vector(self, delta: npt.NDArray):
        for n in range(self.number_bands):
            offset = int(n * len(delta) / self.number_bands)
            self.data.loc[:, f"delta_{n}"] = delta[offset : offset + len(self.k_points)]

    def save(self, path):
        self.data.to_hdf(path, key="table", format="table", data_columns=True)

    @property
    def as_1d_vector(self) -> npt.NDArray:
        return np.concatenate(
            [
                np.array(self.data.loc[:, f"delta_{n}"].values)
                for n in range(self.number_bands)
            ]
        )
