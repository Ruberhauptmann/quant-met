import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.decorators import image_comparison

from quant_met import plotting


@image_comparison(
    baseline_images=["scatter_into_bz"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_scatter_into_bz():
    lattice_constant = np.sqrt(3)

    all_K_points = (
        4
        * np.pi
        / (3 * lattice_constant)
        * np.array(
            [
                (np.sin(i * np.pi / 6), np.cos(i * np.pi / 6))
                for i in [1, 3, 5, 7, 9, 11]
            ]
        )
    )

    plotting.scatter_into_bz(bz_corners=all_K_points, k_points=[[0, 0]])
