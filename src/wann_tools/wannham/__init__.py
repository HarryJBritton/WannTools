from .wannham import WannHam
from wann_tools.config import AppConfig

import numpy as np

__all__ = ['WannHam']

def build_hamiltonian(config: AppConfig):
    """Construct the Hamiltonian described by the config."""

    data_types = {'64': np.complex64,
                  '128': np.complex128}


    ham = WannHam(
        str(config.hamiltonian.hr_file),
        str(config.hamiltonian.nnkp_file),
        dtype=data_types[str(config.hamiltonian.dtype)],
    )

    finite = config.hamiltonian.finite
    if finite.enabled:
        ham.to_finite(layers=finite.layers) # type: ignore

    return ham