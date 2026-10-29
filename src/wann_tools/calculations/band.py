import matplotlib as mpl
import matplotlib.colors
from matplotlib.collections import LineCollection
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator

from wann_tools.calculations.base import ResourceEstimate
from wann_tools.config import AppConfig
from wann_tools.context import CalculationContext
from wann_tools.kspace import kpath
from wann_tools.kspace.space_groups import Space_Groups

from .base import Calculation
from .registry import register_calculation

import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


@register_calculation
class BandCalculation(Calculation):
    name = 'band'

    def validate(self, config: AppConfig, context) -> None:
        ...

    def estimate_resources(self, config: AppConfig, context: CalculationContext) -> ResourceEstimate:
        ...

    def run(self, config: AppConfig, context: CalculationContext, executor):

        settings = config.calculation.band

        # Flatten the k-grid while preserving its original shape.
        k_grid = context.kpoints.k
        original_shape = k_grid.shape[:-1]
        flat_k = k_grid.reshape(-1, 3)

        nk = len(flat_k)
        chunk_size = config.execution.chunk_size
        n_chunks = (nk + chunk_size - 1) // chunk_size

        # Optional debug limit for quick smoke tests.
        if (
            config.debug.enabled
            and config.debug.limit_chunks is not None
        ):
            n_chunks = min(n_chunks, config.debug.limit_chunks)
            logger.warning(
                "Debug chunk limit active: processing %d chunks; "
                "results will be incomplete if chunks were skipped.",
                n_chunks,
            )

        def worker(chunk_id):

            start = chunk_id * chunk_size
            stop = min(start + chunk_size, nk)

            if config.debug.enabled and config.debug.log_chunks:
                logger.debug(
                    "[rank %s] Starting chunk %d/%d, k-points [%d:%d]",
                    getattr(executor, "rank", 0),
                    chunk_id + 1,
                    n_chunks,
                    start,
                    stop,
                )

            t0 = time.perf_counter()

            # Transform only this chunk of k-points.
            hk = context.hamiltonian.ft(flat_k[start:stop])

            nl = (
                config.hamiltonian.finite.layers
                if config.hamiltonian.finite.enabled
                else 1
            )

            chunk_result, chunk_projection = Band(
                hk=hk,
                ef=config.hamiltonian.fermi_energy,
                sp=settings.spin_projection,
            )

            if (
                config.debug.enabled
                and config.debug.log_chunk_timing
            ):
                logger.debug(
                    "[rank %s] Finished chunk %d/%d in %.3f s",
                    getattr(executor, "rank", 0),
                    chunk_id + 1,
                    n_chunks,
                    time.perf_counter() - t0,
                )

            return np.dstack((chunk_result, chunk_projection))

        chunk_results = executor.map_chunks(worker, n_chunks)


        if getattr(executor, 'rank', 0) != 0:
            return None

        if not chunk_results:
            raise RuntimeError("No band chunks were computed")
    
        # Reassemble results in k-point order.
        flat_result = np.concatenate(chunk_results, axis=0)

        # A limited debug run intentionally has fewer k-points.
        if flat_result.shape[0] != nk:
            logger.warning(
                "Partial result: computed %d of %d k-points",
                flat_result.shape[0],
                nk,
            )
            return flat_result

        res_shape = flat_result.shape[-1]

        result = flat_result.reshape(
            original_shape + (context.hamiltonian_dimension, res_shape)
        )
 

        self._write_outputs(config, context, result)
        return result

    def _write_outputs(self, config, context, result):
        output_dir = config.output.directory
        output_dir.mkdir(parents=True, exist_ok=True)

        if config.output.save_data:
            np.save(output_dir / "energies.npy", result)

        if config.output.plot:
            if config.kspace.type == 'surface':
                self._plot_surface(config,  context.kpoints, result)

            if config.kspace.type == 'path':
                self._plot_path(config, context.kpoints, result, config.calculation.band.spin_projection)


    @staticmethod
    def _plot_surface(config, kspace, result):
        logging.getLogger("matplotlib").setLevel(logging.WARNING)


        energy = result[..., 0]
        proj_x = result[..., 1]
        proj_y = result[..., 2]
        proj_z = result[..., 3]

        SCALE = 4

        # Pre config
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams.update({'font.size': 7 * SCALE})
        fig, ax = plt.subplots(figsize=(2 * SCALE, 2 * SCALE))
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        cmap = plt.get_cmap('coolwarm', 256*64)
        ax_lw = 0.5 * SCALE
        band_lw = 0.6 * SCALE

        X = kspace.k[..., 0] - (1.8292726 ) / 3.
        Y = kspace.k[..., 1] - (2.1122620+ 1.0561310) / 3.

        # ax.pcolormesh(X,Y, proj_z[..., 41], cmap='bwr')

        for ri in range(energy.shape[-1]):

            # if ri != 40: continue

            level = 0.75
            cs = ax.contour(X,Y, energy[..., ri], levels=[level])
            if energy[..., ri].max() >level and energy[..., ri].min() < level:
                print(f'band {ri}: Max = {energy[..., ri].max()} Min = {energy[..., ri].min()}')


                interp_u = RegularGridInterpolator(
                    (Y[:, 0], X[0,:]), proj_x[...,ri],
                    bounds_error=False,
                    fill_value=np.nan
                )
                interp_v = RegularGridInterpolator(
                    (Y[:, 0], X[0,:]), proj_y[...,ri],
                    bounds_error=False,
                    fill_value=np.nan
                )

                for idx, segments in enumerate(cs.allsegs):
                    for points in segments:
                        if len(points) < 2:
                            continue

                        xc = points[:, 0]
                        yc = points[:, 1]

                        # Evaluate vector field at contour points
                        query_points = np.column_stack((yc, xc))
                        uc = interp_u(query_points)
                        vc = interp_v(query_points)

                        # Optional: normalize vectors to unit length
                        mag = np.hypot(uc, vc)
                        mag[mag == 0] = 1
                        uc = uc / mag
                        vc = vc / mag


                        ax.quiver(
                            xc[::4], yc[::4],
                            uc[::4], vc[::4],
                            angles="xy",
                            scale_units="xy",
                            scale=50,
                            color="blue",
                            width=0.008
                        )



        # for key, val in Space_Groups['160'].items():
        #     pos = kspace.bvec @ val

        #     ax.scatter(pos[0], pos[1], color='black', marker='X')
        #     ax.annotate(key, xy=pos[:2])

        ax.scatter(0, 0, color='black', marker='x')
        # ax.annotate('K', xy=(0,0))

        ax.set_xlim((X.min(), X.max()))
        ax.set_ylim((Y.min(), Y.max()))

        ax.set_ylabel(r'$k_y$ ($\AA^{-1}$)')
        ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)')

        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

        ax.tick_params(
            which='major',
            width=ax_lw,
            length=3*SCALE,
            labelsize=5*SCALE,
        )
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(
            which='minor',
            width=ax_lw,
            length=1*SCALE,
        )
        ax.set_aspect('equal', adjustable='box')



        fig.savefig(
            config.output.directory / "surface_bands.pdf",
            bbox_inches="tight",
            # pad_inches=0,
            dpi=100,
        )
        plt.close(fig)


    
    @staticmethod
    def _plot_path(config, kspace, result, proj_dir=None):
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        proj = result[..., 1]
        result = result[...,0]

        SCALE = 4

        # Pre config
        plt.rcParams['font.family'] = 'Arial'
        px =1 # 1/plt.rcParams['figure.dpi']
        plt.rcParams.update({'font.size': 7 * SCALE})
        fig, ax = plt.subplots(figsize=(2 * SCALE, 1.2 * SCALE))

        if config.calculation.band.colour is None:
            colour = None
        else:
            colour = matplotlib.colors.CSS4_COLORS[
                config.calculation.band.colour
            ]

        # TODO combine the colour and cmap options 
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        cmap = plt.get_cmap('coolwarm', 256*64)
        
        
        ax_lw = 0.5 * SCALE
        band_lw = 0.6 * SCALE
    
        k = kspace.time
        for ri in range(result.shape[-1]):
            if proj is None:
                ax.plot(
                    k,
                    result[...,ri],
                    color=colour,
                    # marker= '.',
                    # markersize=4,
                    lw= .5,
                )
            else:

                points = np.array([k, result[...,ri]]).T.reshape(-1,1,2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                lc = LineCollection(
                    segments=segments,
                    norm=norm,
                    cmap=cmap,
                )

                lc.set_array((proj[...,ri][:-1]+ proj[...,ri][1:])/2)
                lc.set_linewidth(band_lw)

                ax.add_collection(lc)


        # Colour bar
        if proj is not None:

            pos = ax.get_position() 

            cax = fig.add_axes([
                pos.x1 + 0.03,
                pos.y0,
                0.03,
                pos.height * 0.3,
            ]) # type: ignore

            for spine in cax.spines.values():
                spine.set_linewidth(4*ax_lw)

            cbar = fig.colorbar(
                lc,
                ax=ax,
                cax=cax,
            )

            cbar.set_ticks([-1,0,1])

            cbar.set_label(
                fr'‹$s_{proj_dir if proj_dir is not None else ''}$›',
                rotation=0,
                labelpad=10,
                fontsize=7*SCALE,
            )

            cbar.ax.yaxis.set_label_coords(0.7, 1.5)
            cbar.ax.tick_params(
                width=ax_lw,
                length=3*SCALE,
                labelsize=5*SCALE,
            )
            


        ax.set_ylabel(r'$E - E_F$ (eV)')
        ax.set_xlim((0,1))
        ax.set_ylim((
            config.calculation.band.energy_min,
            config.calculation.band.energy_max,
        ))
        latex_path = [r'$\Gamma$' if p=='G' else p for p in kspace.path]

        # Ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_xticks(
            ticks=kspace.time_labels,
            labels=latex_path,
        )
        ax.tick_params(
            which='major',
            width=ax_lw,
            length=3*SCALE,
            labelsize=5*SCALE,
        )
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(
            which='minor',
            width=ax_lw,
            length=1*SCALE,
        )

        # Axis lw
        for spine in ax.spines.values():
            spine.set_linewidth(ax_lw)

        ax.axhline(0,color='black', lw =ax_lw)
        for pi in range(len(kspace.path)):
            ax.axvline(
                x=kspace.time_labels[pi],
                color='black',
                lw=ax_lw,
            )

        fig.savefig(
            config.output.directory / "path_bands.pdf",
            bbox_inches="tight",
            # pad_inches=0,
            dpi=100,
        )
        plt.close(fig)

def Band(hk, ef: float = 0, sp: str | None = None):

    nb = hk.shape[-1]
    no = nb // 2

    proj_op = {
        'x': (np.eye(N=nb, k=1) + np.eye(N=nb, k=-1)),
        'y': 1j*(-np.eye(N=nb, k=1) + np.eye(N=nb, k=-1)),
        'z': np.diag([1, -1] * no),
    }


    vals, vecs = np.linalg.eigh(hk)
    if sp is None or sp =='0':
        proj = np.ones_like(vals)
    elif sp in ('x', 'y', 'z'):
        proj =  np.real(np.diagonal(np.einsum('kji, jl, klm-> kim',vecs.conj(), proj_op[sp], vecs), axis1=1, axis2=2))
    elif sp == 'all':
        proj =  np.dstack((
            np.real(np.diagonal(np.einsum('kji, jl, klm-> kim',vecs.conj(), proj_op['x'], vecs), axis1=1, axis2=2)),
            np.real(np.diagonal(np.einsum('kji, jl, klm-> kim',vecs.conj(), proj_op['y'], vecs), axis1=1, axis2=2)),
            np.real(np.diagonal(np.einsum('kji, jl, klm-> kim',vecs.conj(), proj_op['z'], vecs), axis1=1, axis2=2)),
        ))



    return  vals - ef, proj
