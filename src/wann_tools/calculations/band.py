
import logging
import time
from typing import cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import CSS4_COLORS, Colormap, LogNorm, Norm, Normalize, PowerNorm
from matplotlib.ticker import AutoMinorLocator, LinearLocator, MaxNLocator
from scipy.interpolate import LinearNDInterpolator

from wann_tools.calculations.base import ResourceEstimate
from wann_tools.config import AppConfig, BandConfig, OutputConfig, SurfaceConfig
from wann_tools.context import CalculationContext
from wann_tools.executors.base import Executor
from wann_tools.kspace import KArea
from wann_tools.util.w90_orbs import construct_am_op

from .base import Calculation
from .registry import register_calculation

logger = logging.getLogger(__name__)


@register_calculation
class BandCalculation(Calculation):
    name = 'band'

    def validate(self, config: AppConfig, context) -> None:
        ...

    def estimate_resources(
        self,
        config: AppConfig,
        context: CalculationContext
    )-> ResourceEstimate:
        ...

    def run(self,
        config: AppConfig,
        context: CalculationContext,
        executor: Executor
    ) -> int:

        # settings = cast(BandConfig, config.calculation.band)

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

            # nl = (
            #     config.hamiltonian.finite.layers
            #     if config.hamiltonian.finite.enabled
            #     else 1
            # )

            chunk_result, chunk_s_projection, chunk_l_projection = band(
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

            return np.dstack((chunk_result, chunk_s_projection, chunk_l_projection))

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

        # extract settings
        settings = cast(BandConfig, config.calculation.band)
        plot_settings = settings.plot
        surface_settings = cast(SurfaceConfig, config.kspace.surface)
        output_settings = config.output

        # Pre config
        scale = 4
        plt.rcParams['font.family'] = output_settings.font
        plt.rcParams.update({'font.size': output_settings.font_size * scale})
        figsize=tuple(scale * x for x in output_settings.fig_size)
        fig, ax = plt.subplots(figsize=figsize)


        # TODO allow for solid colour
        # Colour set up 
        if plot_settings.clim is not None:
             vmin, vmax = plot_settings.clim 
        else:
             vmin, vmax = (None, None)
        norms = {
            "linear": lambda: Normalize(vmin=vmin, vmax=vmax),
            "sqrt": lambda: PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax),
            "cbrt": lambda: PowerNorm(gamma=1./3., vmin=vmin, vmax=vmax),
            "log": lambda: LogNorm(vmin=None, vmax=vmax),
        }
        norm = norms[plot_settings.scale]()
        cmap = plt.get_cmap(plot_settings.colour, 256*64)
        
        # Main variables
        energy = result[..., 0]
        s_x = result[..., 1]
        s_y = result[..., 2]
        # s_z = result[..., 3]
        l_x = result[..., 4]
        l_y = result[..., 5]
        # l_z = result[..., 6]

        if plot_settings.colour_by is None:
            raise NotImplementedError('Plot without arrows is not implemented yet.')
        else:
            if plot_settings.colour_by == 's':
                plot_x = s_x
                plot_y = s_y
            elif plot_settings.colour_by == 'l':
                plot_x = l_x
                plot_y = l_y
            elif plot_settings.colour_by == 'j':
                plot_x = l_x + 0.5 * s_x
                plot_y = l_y + 0.5 * s_y
            else:
                raise ValueError(
                    f'Invalid colour_by variable {plot_settings.colour_by}'
                )




        ax_lw = 0.5 * scale
        band_lw = 0.6 * scale

        cx = np.mean(kspace.k[..., 0])
        cy = np.mean(kspace.k[..., 1])

        x0 = kspace.k[..., 0] - cx
        y0 = kspace.k[..., 1] - cy

        level = settings.mu
        grid_points = np.column_stack((x0.ravel(), y0.ravel()))

        show_bz = plot_settings.show_BZ
        if show_bz:
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

                cs = ax.contour(
                    x0,
                    y0,
                    energy[..., ri],
                    levels=[level],
                    colors='black',
                    lw = band_lw / 4
                )

                if hexagon is not None:
                    cs.set_clip_path(hexagon)

                interp_u = LinearNDInterpolator(
                    grid_points,
                    plot_x[...,ri].ravel(),
                    fill_value=np.nan
                )
                interp_v  = LinearNDInterpolator(
                    grid_points,
                    plot_y[...,ri].ravel(),
                    fill_value=np.nan
                )

                proj_in_plane = np.sqrt(plot_x[...,ri]**2 + plot_y[...,ri]**2)

                interp_z = LinearNDInterpolator(
                    grid_points,
                    proj_in_plane.ravel(),
                    fill_value=np.nan
                )

                for segments in cs.allsegs:
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

                        if plot_settings.project_dir != "none":

                            if plot_settings.project_dir == "tangent":
                                axis_x = -yc.copy()
                                axis_y = xc.copy()

                            elif plot_settings.project_dir == "perpendicular":
                                axis_x = xc.copy()
                                axis_y = yc.copy()

                            else:
                                raise ValueError(
                                    f"Unknown projection: {plot_settings.project_dir}"
                                )

                            # Normalize projection axis
                            normal = np.sqrt(axis_x**2 + axis_y**2)
                            axis_x /= normal
                            axis_y /= normal

                            # Project vector field onto selected axis
                            dot = uc * axis_x + vc * axis_y
                            uc = dot * axis_x
                            vc = dot * axis_y

                        valid = np.isfinite(uc) & np.isfinite(vc)

                        logger.debug(
                            f"Contour points: {len(points)}, "
                            f"valid interpolations: {valid.sum()}"
                        )


                        spacing = plot_settings.arrow_spacing

                        q = ax.quiver(
                            xc[::spacing], yc[::spacing],
                            uc[::spacing], vc[::spacing],
                            zc[::spacing],
                            angles="xy",
                            scale_units="xy",
                            scale=plot_settings.arrow_scale,
                            cmap=cmap,
                            norm=norm,
                            width=0.008
                        )

                        if hexagon is not None:
                            q.set_clip_path(hexagon)

                if plot_settings.show_title:    
                    unit = " meV" if level != 0 else ""
                    ax.set_title(f"$E-E_F = {int(1000 * level):3d}{unit}$")

        for key, value in kspace.space_group.items():

            sym_points = plot_settings.sym_points
            if sym_points != 'all' and key not in sym_points:
                continue

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
                marker='o'
            )
            ax.annotate(
                key if key != 'G' else r'$\Gamma$',
                xy=(point[0]-cx,point[1]-cy),
                xytext=(5, 5),
                textcoords='offset points',
            )

        if show_bz:

            radius = np.linalg.norm(np.einsum(
                'ji, j->i',
                kspace.bvec,
                np.array([0.5,0.5,0]),
                optimize=True,
            ))

            ax.set_xlim((-radius, radius))
            ax.set_ylim((-radius, radius))
        else:
            ax.set_xlim((x0.min(), x0.max()))
            ax.set_ylim((y0.min(), y0.max()))

        ax.set_ylabel(r'$k_y$ (Å$^{\mathrm{-1}}$)')
        ax.set_xlabel(r'$k_x$ (Å$^{\mathrm{-1}}$)')


        # Ticks
        ax.xaxis.set_major_locator(LinearLocator(3))
        ax.yaxis.set_major_locator(LinearLocator(3))
        ax.tick_params(
            which='major',
            width=ax_lw,
            length=3*scale,
            labelsize=5*scale,
        )
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(
            which='minor',
            width=ax_lw,
            length=1*scale,
        )       

        # Axis lw
        for spine in ax.spines.values():
            spine.set_linewidth(ax_lw)

        if show_bz :
            ax.set_axis_off()
        elif not plot_settings.plot_ax:
            ax.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False
            )

            ax.set_ylabel('')
            ax.set_xlabel('')
            
        else:
            ax.set_axis_on()

        config.output.directory.mkdir(parents=True, exist_ok=True)

        fname = "surface_bands"
        if surface_settings.centre is not None:
            fname += '_' + surface_settings.centre

        fig.savefig(
            config.output.directory / (fname + '.' + output_settings.format),
            bbox_inches="tight",
            dpi=100,
        )
        plt.close(fig)

        if plot_settings.export_CB:
            export_colourbar(
                output_settings=output_settings,
                cmap=cmap,
                norm=norm,
            )
        
    @staticmethod
    def _plot_path(config: AppConfig, kspace, result):
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        # get settings from config
        settings = cast(BandConfig, config.calculation.band)
        plot_settings = settings.plot
        output_settings = config.output
        proj_dir = settings.spin_projection

        # Pre config
        scale = 4 # Figure out how scale will work
        plt.rcParams['font.family'] = output_settings.font
        plt.rcParams.update({'font.size': output_settings.font_size * scale})
        figsize = tuple(scale * x for x in output_settings.fig_size)
        fig, ax = plt.subplots(figsize=figsize) 

        # Colour set up
        if plot_settings.colour is None:
            colour = None
            norm = None
            cmap = None
        elif proj_dir is None:
            colour = CSS4_COLORS[
                plot_settings.colour
            ]
            norm = None
            cmap = None
        else:
            colour = None
            norm = mpl.colors.Normalize(vmin=-1, vmax=1) # type: ignore
            cmap = plt.get_cmap(plot_settings.colour, 256*64)
        
        
        ax_lw = 0.5 * scale
        band_lw = 0.6 * scale
    
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
                    raise ValueError(
                        f"calcualtion.band.spin_projection = '{proj_dir}'"
                        " is not supported for path KStructure."
                    )

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
                fontsize=7*scale,
            )

            cbar.ax.yaxis.set_label_coords(0.7, 1.5)
            cbar.ax.tick_params(
                width=ax_lw,
                length=3*scale,
                labelsize=5*scale,
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
            length=3*scale,
            labelsize=5*scale,
        )
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(
            which='minor',
            width=ax_lw,
            length=1*scale,
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

        config.output.directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            config.output.directory / ("path_bands." + output_settings.format),
            bbox_inches="tight",
            dpi=100,
        )
        plt.close(fig)

def band(hk, ef: float = 0):

    nb = hk.shape[-1]
    no = nb // 2

    spin_proj_op = {
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


    # TODO pass through the orbital configuration
    lx, ly, lz =  construct_am_op(['d','d','d','p','p','p','p','p','p'])
    angular_momentum_op = {
        'x': lx,
        'y': ly,
        'z': lz,
    }

    vals, vecs = np.linalg.eigh(hk)

    s_op = np.array([spin_proj_op['x'], spin_proj_op['y'], spin_proj_op['z']])
    s_proj =  np.real(np.einsum(
        'xik, aij, xjk -> xka',
        vecs.conj(),
        s_op,
        vecs,
        optimize=True,
    ))


    l_op = np.array(
        [angular_momentum_op['x'], angular_momentum_op['y'], angular_momentum_op['z']])
    l_proj =  np.real(np.einsum(
        'xik, aij, xjk -> xka',
        vecs.conj(),
        l_op,
        vecs,
        optimize=True,
    ))


    return  vals - ef, s_proj, l_proj


def export_colourbar(
    output_settings: OutputConfig,
    cmap: Colormap,
    norm: Norm
):

    fig, ax = plt.subplots(figsize=(1.5,4))

    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
    )

    cbar.ax.set_yscale('linear')
    fig.savefig(output_settings.directory / ( 'colourbar.'+ output_settings.format),
                bbox_inches="tight",
                dpi=100,)
    plt.close(fig)



        
    
