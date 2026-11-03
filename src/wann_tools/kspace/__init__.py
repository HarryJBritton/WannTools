from .kpath import KPath
from .karea import KArea
from .load_nnkp import load_nnkp

from wann_tools.config import AppConfig, PathConfig, SurfaceConfig

from typing import cast


def build_kspace(config: AppConfig):
    """Construct the selected k-space geometry."""

    bvec = load_nnkp(str(config.hamiltonian.nnkp_file))

    if config.kspace.type == "surface":
        settings = cast(SurfaceConfig, config.kspace.surface)
        return KArea(
            bvec=bvec,
            n1=settings.n1,
            n2=settings.n2,
            level=settings.level, 
            fixed_axis=settings.fixed_axis,
            centre=settings.centre,
            width=settings.width,
        )
    elif config.kspace.type == 'path':
        settings = cast(PathConfig, config.kspace.path)
        return KPath(
            path=settings.path, 
            bvec=bvec,
            num_per_section=settings.nps,
            space_group=settings.space_group 
        )

    # Add KVolume here when their APIs are defined.
    raise NotImplementedError(
        f"K-space type {config.kspace.type!r} is not implemented yet"
    )
