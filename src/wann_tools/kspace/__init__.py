from .kpath import KPath
from .karea import KArea
from .load_nnkp import load_nnkp

from wann_tools.config import AppConfig

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
