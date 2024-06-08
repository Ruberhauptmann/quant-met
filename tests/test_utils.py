import numpy as np
import pytest

from quant_met import utils


def test_generate_uniform_grid():
    with pytest.raises(ValueError):
        utils.generate_uniform_grid(
            ncols=1,
            nrows=10,
            corner_1=np.array([0, 1]),
            corner_2=np.array([1, 0]),
            origin=np.array([0, 0]),
        )
    with pytest.raises(ValueError):
        utils.generate_uniform_grid(
            ncols=10,
            nrows=1,
            corner_1=np.array([0, 1]),
            corner_2=np.array([1, 0]),
            origin=np.array([0, 0]),
        )
    with pytest.raises(ValueError):
        utils.generate_uniform_grid(
            ncols=10,
            nrows=10,
            corner_1=np.array([0, 0]),
            corner_2=np.array([1, 0]),
            origin=np.array([0, 0]),
        )
    with pytest.raises(ValueError):
        utils.generate_uniform_grid(
            ncols=10,
            nrows=10,
            corner_1=np.array([1, 0]),
            corner_2=np.array([0, 0]),
            origin=np.array([0, 0]),
        )
    # print(grid)
    # assert False


# Add hypothesis test!
