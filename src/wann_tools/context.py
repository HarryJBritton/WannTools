from dataclasses import dataclass
from typing import Any

from wann_tools.config import AppConfig

import numpy as np

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


def build_context(config: AppConfig, ham: Any, kspace: Any):

    # Hamiltonian metadata
    hamiltonian_dimension = ham.nb
    # Use the actual dtype if the Hamiltonian exposes it.
    hamiltonian_dtype = str(
        getattr(ham, "dtype", np.dtype(np.complex128))
    )

    # kspace data
    kpoints = kspace.k
    total_kpoints = int(np.prod(kpoints.shape[:-1]))

    return  CalculationContext(
        hamiltonian_dimension=hamiltonian_dimension,
        hamiltonian_dtype=hamiltonian_dtype,
        total_kpoints=total_kpoints,
        hamiltonian=ham,
        kpoints=kspace,
        kspace_type=config.kspace.type,
        finite_layers=(
            config.hamiltonian.finite.layers
            if config.hamiltonian.finite.enabled
            else None
        ),
    )