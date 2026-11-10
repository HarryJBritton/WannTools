from wann_tools.kspace.base import KSpaceProvider

from .kpath import KPath
from .karea import KArea
from .load_nnkp import load_nnkp

from wann_tools.config import AppConfig, KSpaceConfig, PathConfig, SurfaceConfig

from typing import cast


def build_kspace(config: AppConfig) -> KSpaceProvider:
    """Construct the selected k-space geometry."""

    bvec = load_nnkp(str(config.hamiltonian.nnkp_file))

    kspace_settings = cast(KSpaceConfig, config.kspace)

    if config.kspace.type == "surface":
        area_settings = cast(SurfaceConfig, config.kspace.surface)
        return KArea(
            bvec=bvec,
            group=kspace_settings.space_group,
            n1=area_settings.n1,
            n2=area_settings.n2,
            level=area_settings.level, 
            fixed_axis=area_settings.fixed_axis,
            centre=area_settings.centre,
            width=area_settings.width,
        )
    elif config.kspace.type == 'path':
        path_settings = cast(PathConfig, config.kspace.path)
        return KPath(
            bvec=bvec,
            group=kspace_settings.space_group,
            path=path_settings.path,
            num_per_section=path_settings.nps,
        )

    # Add KVolume here when their APIs are defined.
    raise NotImplementedError(
        f"K-space type {config.kspace.type!r} is not implemented yet"
    )
