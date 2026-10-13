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

    hamiltonian_storage_bytes: int = 0


def build_context(config: AppConfig, ham: Any, kspace: Any):

    # Hamiltonian metadata
    hamiltonian_dimension = ham.nb
    # Use the actual dtype if the Hamiltonian exposes it.
    hamiltonian_dtype = ham.dtype
    

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
        hamiltonian_storage_bytes=array_storage_bytes(
            ham.hr,
            ham.rvec,
            ham.ndeg,
        ),
    )

def array_storage_bytes(*arrays) -> int:
    return sum(
        array.nbytes
        for array in arrays
        if array is not None
    )