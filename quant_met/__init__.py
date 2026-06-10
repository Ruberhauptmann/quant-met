# 1. Import everything from your Python modules
from .geometries import decorated_graphene_lattice, square_lattice

# 2. Import your compiled C++ core
try:
    from . import quant_met_core as core
except ImportError as e:
    raise ImportError(
        "Failed to import the compiled C++ core. Ensure the project was "
        "installed using 'pip install -e .'"
    ) from e

# 3. Expose a unified API to the user
__all__ = ["decorated_graphene_lattice", "square_lattice", "core"]
