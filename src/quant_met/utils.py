import numpy as np
import numpy.typing as npt


def generate_uniform_grid(
    ncols: int,
    nrows: int,
    corner_1: npt.NDArray[np.float64],
    corner_2: npt.NDArray[np.float64],
    origin: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    grid: npt.NDArray[np.float64] = np.concatenate(
        [
            np.linspace(
                origin[0] + i / (nrows - 1) * corner_2,
                origin[1] + corner_1 + i / (nrows - 1) * corner_2,
                num=ncols,
            )
            for i in range(nrows)
        ]
    )

    return grid
