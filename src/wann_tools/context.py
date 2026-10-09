from dataclasses import dataclass
from typing import Any


@dataclass
class CalculationContext:
    # Metadata needed for planning and estimation.
    hamiltonian_dimension: int
    hamiltonian_dtype: str
    total_kpoints: int

    # Shared objects or providers used at execution time.
    hamiltonian: Any = None
    kpoints: Any = None

    # Geometry and model metadata.
    kspace_type: str = ""
    finite_layers: int | None = None
    orbitals_per_cell: int | None = None

    