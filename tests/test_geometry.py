# SPDX-FileCopyrightText: 2024 Tjark Sievers
#
# SPDX-License-Identifier: MIT

import numpy as np
from quant_met import geometry

def test_generate_bz_path():
    graphene_lattice = geometry.Graphene(lattice_constant=np.sqrt(3))
    band_path, band_path_plot, ticks, labels = graphene_lattice.generate_high_symmetry_path(
        number_of_points=1000
    )

    assert labels == ["$M$", "$\\Gamma$", "$K$", "$M$"]
    assert ticks[0] == 0.0
    assert band_path_plot[0] == 0.0
