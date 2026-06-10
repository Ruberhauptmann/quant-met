import quant_met as qm
import numpy as np

# 1. Grab your pre-defined Python geometries
lattice_vectors, sublattices = qm.decorated_graphene_lattice()
print("Loaded geometry vectors:", lattice_vectors)

# 2. Feed data right into your high-performance C++ backend
test_matrix = np.eye(4, dtype=np.complex128)
scaled = qm.core.scale_matrix_parallel(test_matrix, 3.14)

print("\nResult from parallel C++ core:\n", scaled)
