import numpy as np


def decorated_graphene_lattice():
    """
    Returns the basis vectors and atomic positions for
    a decorated graphene layout.
    """
    # Primitive lattice vectors
    a1 = np.array([3 / 2, np.sqrt(3) / 2])
    a2 = np.array([3 / 2, -np.sqrt(3) / 2])

    # Sublattice positions (A, B, decorated sites, etc.)
    positions = {
        "A": np.array([0, 0]),
        "B": np.array([1, 0]),
    }

    return [a1, a2], positions


def square_lattice():
    return [np.array([1.0, 0.0]), np.array([0.0, 1.0])], {"A": np.array([0.0, 0.0])}
