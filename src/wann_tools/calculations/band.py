import matplotlib as mpl
import matplotlib.colors
from matplotlib.collections import LineCollection
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt

from wann_tools.calculations.base import ResourceEstimate
from wann_tools.config import AppConfig
from wann_tools.context import CalculationContext

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

        result = flat_result[...,0].reshape(
            original_shape + (context.hamiltonian_dimension,)
        )
        projection = flat_result[...,1].reshape(
            original_shape + (context.hamiltonian_dimension,)
        )

        if settings.spin_projection in (None, '0'):
            projection = None
        

        self._write_outputs(config, context, result, projection)
        return result

    def _write_outputs(self, config, context, result, projection):
        output_dir = config.output.directory
        output_dir.mkdir(parents=True, exist_ok=True)

        if config.output.save_data:
            np.save(output_dir / "energies.npy", result)

        if config.output.plot:
            if config.kspace.type == 'surface':
                raise NotImplementedError

            if config.kspace.type == 'path':
                self._plot_path(config, context.kpoints, result, projection)

    
    @staticmethod
    def _plot_path(config, kspace, result, proj=None):
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        # Pre config
        plt.rcParams['font.family'] = 'Arial'
        px = 1/plt.rcParams['figure.dpi']
        plt.rcParams.update({'font.size': 6.6})
        fig, ax = plt.subplots(figsize=(1660 * px * 1., 1000* px))

        if config.calculation.band.colour is None:
            colour = None
        else:
            colour = matplotlib.colors.CSS4_COLORS[
                config.calculation.band.colour
            ]

        # TODO combine the colour and cmap options 
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        cmap = plt.get_cmap('coolwarm', 256*64)
        
        
        ax_lw = 0.5    
    
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
                lc.set_linewidth(0.5)

                ax.add_collection(lc)


        # Colour bar
        if proj is not None:

            pos = ax.get_position() 

            cax = fig.add_axes([
                pos.x1 + 0.02,
                pos.y0,
                0.05,
                pos.height * 0.3,
            ]) # type: ignore

            for spine in cax.spines.values():
                spine.set_linewidth(ax_lw)

            cbar = fig.colorbar(
                lc,
                ax=ax,
                cax=cax,
            )

            cbar.set_label(
                fr'‹$s_{"y"}$›',
                rotation=0,
                labelpad=10
            )

            cbar.ax.yaxis.set_label_coords(0.6, 1.5)
            cbar.ax.tick_params(width=ax_lw)
            


        ax.set_ylabel(r'$E - E_F$ (eV)')
        ax.set_xlim((0,1))
        ax.set_ylim((
            config.calculation.band.energy_min,
            config.calculation.band.energy_max,
        ))
        latex_path = [r'$\Gamma$' if p=='G' else p for p in kspace.path]

        # Ticks
        ax.set_xticks(
            ticks=kspace.time_labels,
            labels=latex_path,
        )
        ax.tick_params(width=ax_lw)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

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
            dpi=300,
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
    # TODO implement X and Y projection
    if sp is None or sp =='0':
        proj = np.ones_like(vals)
    else:
        proj =  np.real(np.diagonal(np.einsum('kji, jl, klm-> kim',vecs.conj(), proj_op[sp], vecs), axis1=1, axis2=2))

    return  vals - ef, proj
