from wann_tools.calculations.base import Calculation
from wann_tools.config import AppConfig, load_config
from wann_tools.wannham import WannHam
from wann_tools.kspace import KPath, KArea, load_nnkp
from wann_tools.calculations import spectral  # Registers spectral
from wann_tools.calculations.registry import get_calculation
from wann_tools.context import CalculationContext
from wann_tools.executor import build_executor

import numpy as np

def build_hamiltonian(config: AppConfig):
    """Construct the Hamiltonian described by the config."""

    data_types = {'64': np.complex64,
                  '128': np.complex128}
    ham = WannHam(
        str(config.hamiltonian.hr_file),
        dtype=data_types[str(config.hamiltonian.dtype)],
    )

    finite = config.hamiltonian.finite
    if finite.enabled:
        ham.to_finite(layers=finite.layers) # type: ignore

    return ham


def build_kspace(config: AppConfig):
    """Construct the selected k-space geometry."""

    bvec = load_nnkp(str(config.hamiltonian.nnkp_file))

    if config.kspace.type == "surface":
        settings = config.kspace.surface
        return KArea(
            bvec=bvec,
            n1=settings.n1, # type: ignore
            n2=settings.n2, # type: ignore
            level=settings.level, # type: ignore
            fixed_axis=settings.fixed_axis, # type: ignore
        )
    elif config.kspace.type == 'path':
        settings = config.kspace.path
        return KPath(
            path=settings.path, # type: ignore
            bvec=bvec,
            num_per_section=settings.nps, # type: ignore
        )

    # Add KVolume here when their APIs are defined.
    raise NotImplementedError(
        f"K-space type {config.kspace.type!r} is not implemented yet"
    )


def run(config: AppConfig):
    ham = build_hamiltonian(config)
    kspace = build_kspace(config)

    calculation = get_calculation(config.calculation.type)

    # Hamiltonian metadata
    hamiltonian_dimension = ham.nb
    # Use the actual dtype if the Hamiltonian exposes it.
    hamiltonian_dtype = str(
        getattr(ham, "dtype", np.dtype(np.complex128))
    )

    # Count k-points without assuming the final coordinate axis is a
    # spatial dimension beyond the conventional (..., 3) layout.
    kpoints = kspace.k
    total_kpoints = int(np.prod(kpoints.shape[:-1]))

    context = CalculationContext(
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

    calculation.validate(
        config=config,
        context=context,
    )

    executor = build_executor(config.execution)

    return calculation.run(
        config=config,
        context=context,
        executor=executor,
    )

if __name__=='__main__':
    config=load_config('config.yaml')
    run(config=config)