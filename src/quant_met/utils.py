import numpy as np


def generate_uniform_grid(ncols, nrows, corner_1, corner_2, origin):
    grid = np.concatenate(
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
