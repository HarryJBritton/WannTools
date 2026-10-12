from wann_tools.calculations.base import ResourceEstimate
from wann_tools.config import AppConfig

from .base import Calculation
from .registry import register_calculation

import numpy as np
import matplotlib.pyplot as plt
import logging
import time

logger = logging.getLogger(__name__)

@register_calculation
class SpectralCalculation(Calculation):
    name = 'spectral'

    def validate(self, config: AppConfig, context) -> None:
        ...
    
    def estimate_resources(self, config: AppConfig, context) -> ResourceEstimate:
        ...
    
    def run(self, config: AppConfig, context, executor):
        settings = config.calculation.spectral

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

        omega = settings.omega.to_array()

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

            chunk_result = Spectral(
                hk=hk,
                omega=omega,
                sproj=settings.spin_projection,
                lproj=settings.layer_projection,
                nl=nl,
                eta=settings.eta,
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

            return chunk_result

        # This is where the executor actually distributes the work.
        chunk_results = executor.map_chunks(worker, n_chunks)

        # MPIExecutor returns the complete list only on rank 0.
        if getattr(executor, "rank", 0) != 0:
            return None

        if not chunk_results:
            raise RuntimeError("No spectral chunks were computed")

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

        result = flat_result.reshape(
            original_shape + (len(omega),)
        )

        self._write_outputs(config, context, result, omega)
        return result


    def _write_outputs(self, config, context, result, omega):
        output_dir = config.output.directory
        output_dir.mkdir(parents=True, exist_ok=True)

        if config.output.save_data:
            np.save(output_dir / "spectral.npy", result)
            np.save(output_dir / "energies.npy", omega)

        if config.output.plot:
            if config.kspace.type == 'surface':
                self._plot_surface(config, context.kpoints, result)

            if config.kspace.type == 'path':
                self._plot_path(config, context.kpoints, result, omega)


    @staticmethod
    def _plot_surface(config, kspace, result):
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(
            kspace.k[:, :, 0],
            kspace.k[:, :, 1],
            result[:, :, 0],
            cmap="bwr",
            vmin=-0.2,
            vmax=0.2,
            shading="auto",
        )
        ax.set_aspect("equal")
        fig.colorbar(mesh, ax=ax, label="Spectral function")

        fig.savefig(
            config.output.directory / "surface_spec.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


    @staticmethod
    def _plot_path(config, kspace, result, omega):
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        K, O = np.meshgrid(kspace.length, omega, indexing='ij')

        

        
        fig, ax = plt.subplots(figsize=(10,8))

        if config.calculation.spectral.spin_projection == '0':
            cmap= 'bone'
            vmin= 0
            vmax= result.max()
        else:
            cmap= 'bwr'
            vmin= -0.5
            vmax= 0.5

        mesh = ax.pcolormesh(
            K,
            O - 8.11,
            result[:, :],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.set_ylabel('Energy (meV)')
        ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)')
        fig.colorbar(mesh, ax=ax, label="Spectral function")

        fig.savefig(
            config.output.directory / "path_spec.png",
            bbox_inches="tight",
        )
        plt.close(fig)




def Spectral(hk, omega, sproj=None, eta=0.05, lproj=None, nl=1):

    pauli = {
            '0': np.array([[1,0],[0,1]], dtype=np.complex64),
            'x': np.array([[0,1],[1,0]], dtype=np.complex64),
            'y': np.array([[0,-1j],[1j,0]], dtype=np.complex64),
            'z': np.array([[1,0],[0,-1]], dtype=np.complex64),
        }

    k_shape = hk.shape[:-2]
    nkp = np.prod(k_shape)
    nb = hk.shape[-1]
    no = nb // (2 * nl)

    # print(f'{nb} bands across {nl} layers\nwith {no} orbitals and {2} spins.')


    flat_hk = hk.reshape((nkp,nb,nb))

    # Spin projection
    if sproj is None:
        ps = np.eye(2)
    elif sproj not in pauli.keys():
        raise Exception(f'spin projection {sproj} is not a Pauli matrix.\nMust be in {pauli.keys()}')
    else:
        ps = pauli[sproj]

    # TODO orbital projection
    po = np.eye(no)

    # Layer projection
    if lproj is None:
        pl = np.eye(nl)
    elif lproj not in range(nl):
        raise Exception('layer projection onto layer {lproj} is not valid for {nl} layer system')
    else:
        pl = np.zeros((nl, nl))
        pl[lproj, lproj] = 1


    total_proj = np.kron(pl, np.kron(po, ps))

    E, V = np.linalg.eigh(flat_hk)

    flat_A = np.zeros((nkp, len(omega)))
    for ik in range(nkp):

        G = 1/(1j*eta + omega[None,:] - E[ik, :,None])

        flat_A[ik,:] = -1 / np.pi * np.imag(np.einsum('iw, ai, ab, bi -> w',G, V[ik].conj(), total_proj, V[ik],  optimize=True))
        
    return flat_A.reshape(k_shape+(len(omega),))





