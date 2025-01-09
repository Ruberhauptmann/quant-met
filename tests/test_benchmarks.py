# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

from statistics import median

import pytest


@pytest.mark.benchmark()
def test_median_performance():
    return median([1, 2, 3, 4, 5])
