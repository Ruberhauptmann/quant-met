import pytest
import sisl

@pytest.fixture
def square_lattice_tb():
    """Create a square lattice tight-binding model using sisl's close() style."""
    a = 1.0        # Lattice constant
    t = -1.0       # Nearest-neighbor hopping
    bond = a       # Bond length

    # Create an atom object with appropriate cutoff range
    atom = sisl.Atom(1, R=bond + 0.01)

    # Generate 2D square lattice geometry
    geom = sisl.Geometry(
        [[0., 0., 0.]],           # Single atom at origin
        [atom],
        [[a, 0., 0.], [0., a, 0.], [0., 0., 10.]],  # Unit cell (2D in 3D space)
    )

    hamiltonian = sisl.Hamiltonian(geom)
    search_radius = [0.1 * bond, bond + 0.01]  # Search radius for neighbors

    for ia in geom:
        idx_a = geom.close(ia, R=search_radius)
        hamiltonian[ia, idx_a[0]] = 0.0  # On-site energy
        for i in idx_a[1:]:
            hamiltonian[ia, i] = t      # Nearest-neighbor hopping

    hamiltonian.finalize()
    yield hamiltonian
