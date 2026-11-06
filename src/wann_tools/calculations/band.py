import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm, Norm, Normalize, CSS4_COLORS
from matplotlib.collections import LineCollection
from matplotlib.ticker import AutoMinorLocator, LinearLocator, MaxNLocator

from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from typing import cast


from wann_tools.calculations.base import ResourceEstimate
from wann_tools.config import AppConfig, BandConfig, SurfaceConfig
from wann_tools.context import CalculationContext
from wann_tools.executors.base import Executor
from wann_tools.kspace import kpath, KArea, KPath
from wann_tools.kspace.space_groups import Space_Groups

from .base import Calculation
from .registry import register_calculation

import numpy as np
import logging
import time
import csv


logger = logging.getLogger(__name__)


@register_calculation
class BandCalculation(Calculation):
    name = 'band'

    def validate(self, config: AppConfig, context) -> None:
        ...

    def estimate_resources(self, config: AppConfig, context: CalculationContext) -> ResourceEstimate:
        ...

    def run(self, config: AppConfig, context: CalculationContext, executor: Executor) -> int:

        settings = cast(BandConfig, config.calculation.band)

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
            return None # type: ignore

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
            return 1

        res_shape = flat_result.shape[-1]

        result = flat_result.reshape(
            original_shape + (context.hamiltonian_dimension, res_shape)
        )
 

        self._write_outputs(config, context, result)
        return 0

    def _write_outputs(self, config: AppConfig, context: CalculationContext, result):
        output_dir = config.output.directory
        output_dir.mkdir(parents=True, exist_ok=True)

        if config.output.save_data:
            np.save(output_dir / "energies.npy", result)

        if config.output.plot:
            if config.kspace.type == 'surface':
                self._plot_surface(config,  context.kpoints, result)

            if config.kspace.type == 'path':
                self._plot_path(config, context.kpoints, result)


    @staticmethod
    def _plot_surface(config: AppConfig, kspace: KArea, result):
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        settings = cast(BandConfig, config.calculation.band)
        surface_settings = cast(SurfaceConfig, config.kspace.surface)
        output_settings = config.output


        energy = result[..., 0]
        proj_x = result[..., 1]
        proj_y = result[..., 2]
        proj_z = result[..., 3]


        # Pre config
        SCALE = 4
        plt.rcParams['font.family'] = output_settings.font
        plt.rcParams.update({'font.size': output_settings.font_size * SCALE})
        fig, ax = plt.subplots(figsize=tuple(SCALE * x for x in output_settings.fig_size)) 

        # Colour set up
        norm = mpl.colors.Normalize(vmin=-1, vmax=1) # type: ignore
        cmap = plt.get_cmap(settings.colour, 256*64)


        ax_lw = 0.5 * SCALE
        band_lw = 0.6 * SCALE

        cx = np.mean(kspace.k[..., 0])
        cy = np.mean(kspace.k[..., 1])

        X = kspace.k[..., 0] - cx
        Y = kspace.k[..., 1] - cy

        # TODO writeout contour 
        # f = open(output_settings.directory / "contour.csv", "w", newline="")
        # writer = csv.writer(f)

        level = settings.mu
        grid_points = np.column_stack((X.ravel(), Y.ravel()))


        FBZ = False
        if FBZ:
            hexagon = patches.RegularPolygon(
                (0, 0),
                6,
                radius=1.219514942,
                orientation=np.pi / 6,
                transform=ax.transData,
                fill=False,
                edgecolor='black',
                linewidth=band_lw,
                zorder=10
            )
            ax.add_patch(hexagon)
        else:
            hexagon = None



        for ri in range(energy.shape[-1]):
            if energy[..., ri].max() >level and energy[..., ri].min() < level:

                cs = ax.contour(X,Y, energy[..., ri], levels=[level], colors='black', lw = band_lw / 4)
                if hexagon is not None:
                    cs.set_clip_path(hexagon)

                interp_u = LinearNDInterpolator(
                    grid_points,
                    proj_x[...,ri].ravel(),
                    fill_value=np.nan
                )
                interp_v  = LinearNDInterpolator(
                    grid_points,
                    proj_y[...,ri].ravel(),
                    fill_value=np.nan
                )

                # TODO generalise the colouring choice.

                ratio = np.sqrt(proj_x[...,ri]**2 + proj_y[...,ri]**2) / (np.abs(proj_z[...,ri]) + 1e-12)

                interp_z = LinearNDInterpolator(
                    grid_points,
                    ratio.ravel(),
                    fill_value=np.nan
                )

                
                # # Writer for contours
                # for path in cs.get_paths():
                #     vert = path.vertices
                #     writer.writerow(vert) # type: ignore
                #     writer.writerow(["", ""])  # Separate disconnected paths



                for idx, segments in enumerate(cs.allsegs):
                    for points in segments:
                        if len(points) < 2:
                            continue

                        xc = points[:, 0]
                        yc = points[:, 1]

                        # Evaluate vector field at contour points
                        query_points = np.column_stack((xc, yc))
                        uc = interp_u(query_points)
                        vc = interp_v(query_points)
                        zc = interp_z(query_points)

                        valid = np.isfinite(uc) & np.isfinite(vc)

                        logger.debug(
                            f"Contour points: {len(points)}, "
                            f"valid interpolations: {valid.sum()}"
                        )

                        # mag = np.hypot(uc, vc)
                        # idx = np.where(mag ==0)
                        # mag[idx] = 1
                        # uc = uc / mag
                        # vc = vc / mag


                        spacing = 6

                        

                        q = ax.quiver(
                            xc[::spacing], yc[::spacing],
                            uc[::spacing], vc[::spacing],
                            zc[::spacing],
                            angles="xy",
                            scale_units="xy",
                            scale=2, #3
                            cmap=cmap,
                            norm=Normalize(vmin=0,vmax=1), #LogNorm(vmin=0.01, vmax=5),
                            width=0.008
                        )

                        
                        if hexagon is not None:
                            q.set_clip_path(hexagon)
                    

                ax.set_title(f'$E$–$E_F$={int(1000 * level):3d}{' meV' if level != 0 else ''}')

        # f.close()


        for key, value in Space_Groups['160'].items():
            point = np.einsum(
                'ji, j -> i',
                kspace.bvec,
                value,
                optimize=True
            )

            ax.scatter(
                point[0] - cx,
                point[1] - cy,
                color='black',
                marker='x'
            )
            ax.annotate(
                key if key != 'G' else r'$\Gamma$',
                xy=(point[0]-cx,point[1]-cy),
                xytext=(5, 5),
                textcoords='offset points',
            )

        # might not be needed otherwise
        # TODO generalise to other lattice vectors
        if FBZ:
            ax.set_xlim((-1.219514942, 1.219514942))
            ax.set_ylim((-1.219514942, 1.219514942))
        else:
            ax.set_xlim((X.min(), X.max()))
            ax.set_ylim((Y.min(), Y.max()))

        ax.set_ylabel(r'$k_y$ (Å$^{\mathrm{-1}}$)')
        ax.set_xlabel(r'$k_x$ (Å$^{\mathrm{-1}}$)')


        # Ticks
        ax.xaxis.set_major_locator(LinearLocator(3))
        ax.yaxis.set_major_locator(LinearLocator(3))
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
        
        # Colourbar 
        # norm = mpl.colors.Normalize(vmin=0, vmax=1) # type: ignore
        # sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap) # type: ignore
        # sm.set_array([])
        # cb = fig.colorbar(
        #     sm,
        #     ax=ax,
        #     orientation='horizontal',
        #     location='top',
        #     shrink=0.75
        # )

       

        # Axis lw
        for spine in ax.spines.values():
            spine.set_linewidth(ax_lw)

        # TODO make this adaptable
        if FBZ:
            ax.set_axis_off()
        else:
            ax.set_axis_on()

            

        # ax.set_aspect('equal')


        fig.savefig(
            config.output.directory / (f"surface_bands{'' if surface_settings.centre is None else "_"+surface_settings.centre}." + output_settings.format),
            bbox_inches="tight",
            dpi=100,
        )
        plt.close(fig)

    @staticmethod
    def _plot_path(config: AppConfig, kspace, result):
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        # get settings from config
        settings = cast(BandConfig, config.calculation.band)
        output_settings = config.output
        proj_dir = settings.spin_projection

        # Pre config
        SCALE = 4 # Figure out how scale will work
        plt.rcParams['font.family'] = output_settings.font
        plt.rcParams.update({'font.size': output_settings.font_size * SCALE})
        fig, ax = plt.subplots(figsize=tuple(SCALE * x for x in output_settings.fig_size) ) 

        # Colour set up
        if settings.colour is None:
            colour = None
            norm = None
            cmap = None
        elif proj_dir is None:
            colour = CSS4_COLORS[
                settings.colour
            ]
            norm = None
            cmap = None
        else:
            colour = None
            norm = mpl.colors.Normalize(vmin=-1, vmax=1) # type: ignore
            cmap = plt.get_cmap(settings.colour, 256*64)
        
        
        ax_lw = 0.5 * SCALE
        band_lw = 0.6 * SCALE
    
        k = kspace.time
        energy = result[..., 0]
        lc = None

        for ri in range(energy.shape[-1]):
            if proj_dir is None:
                ax.plot(
                    k,
                    energy[...,ri],
                    color=colour,
                    lw= band_lw,
                )
            else:

                if proj_dir == 'x':
                    proj = result[...,ri, 1]
                elif proj_dir == 'y':
                    proj = result[...,ri, 2]
                elif proj_dir == 'z':
                    proj = result[...,ri, 3]
                else:
                    raise ValueError(f"calcualtion.band.spin_projection = '{proj_dir}' is not supported for path KStructure.")

                # Generate plot points for variable colouring
                points = np.array([k, energy[...,ri]]).T.reshape(-1,1,2)
                segments = [segment for segment in np.stack(
                    [points[:-1], points[1:]], axis=1
                )]

                lc = LineCollection(
                    segments=segments,
                    norm=norm,
                    cmap=cmap,
                )

                lc.set_array((proj[:-1]+ proj[1:])/2)
                lc.set_linewidth(band_lw)

                ax.add_collection(lc)


        # Colour bar
        if proj_dir is not None and lc is not None:

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
            settings.energy_min,
            settings.energy_max,
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
            config.output.directory / ("path_bands." + output_settings.format),
            bbox_inches="tight",
            dpi=100,
        )
        plt.close(fig)

def Band(hk, ef: float = 0):

    nb = hk.shape[-1]
    no = nb // 2

    proj_op = {
        'x': np.kron(np.eye(no), np.array([
            [0, 1],
            [1, 0],
        ])),
        'y': np.kron(np.eye(no), np.array([
            [0, -1j],
            [1j, 0],
        ])),
        'z': np.kron(np.eye(no), np.array([
            [1, 0],
            [0, -1],
        ])),
    }
    vals, vecs = np.linalg.eigh(hk)

    sz = proj_op['z']

    if False:
        cuts = np.flatnonzero(np.diff(vals) > 1e-5) + 1
        for group in np.split(np.arange(nb), cuts):
            if len(group) < 2:
                continue

            logger.debug('Found degen')

            v = vecs[:,:, group]



            Sproj = np.einsum(
                'xil, ij, xjk -> xlk',
                v.conj(),
                sz,
                v,
                optimize=True,
            )
            _, U = np.linalg.eigh(Sproj)
            vecs[:, :, group] = np.einsum('xij,xjk->xik', v, U, optimize=True)

    S = np.array([proj_op['x'], proj_op['y'], proj_op['z']])
    proj =  np.real(np.einsum(
        'xik, aij, xjk -> xka',
        vecs.conj(),
        S,
        vecs,
        optimize=True,
    ))


    return  vals - ef, proj


