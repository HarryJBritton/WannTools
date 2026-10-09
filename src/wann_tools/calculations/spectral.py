from wann_tools.calculations.base import ResourceEstimate
from wann_tools.config import AppConfig

from .base import Calculation
from .registry import register_calculation

import numpy as np
import matplotlib.pyplot as plt

@register_calculation
class SpectralCalculation(Calculation):
    name = 'spectral'

    def validate(self, config: AppConfig, context) -> None:
        ...
    
    def estimate_resources(self, config: AppConfig, context) -> ResourceEstimate:
        ...
    
    def run(self, config: AppConfig, context, executor):
        settings = config.calculation.spectral

        ham = context.hamiltonian
        kspace = context.kpoints

        hk = ham.ft(kspace.k)
        omega = np.asarray(settings.energies, dtype=float)

        nl = (
            config.hamiltonian.finite.layers
            if config.hamiltonian.finite.enabled
            else context.finite_layers or 1
        )

        result = Spectral(
            hk=hk,
            omega=omega,
            sproj=settings.spin_projection,
            lproj=settings.layer_projection,
            nl=nl,
            eta=settings.eta,
        )

        output_dir = config.output.directory
        output_dir.mkdir(parents=True, exist_ok=True)

        if config.output.save_data:
            np.save(output_dir / "spectral.npy", result)
            np.save(output_dir / "energies.npy", omega)

        if config.output.plot:
            self._plot_surface(config, kspace, result)

        return result

    @staticmethod
    def _plot_surface(config, kspace, result):
        if config.kspace.type != "surface":
            raise NotImplementedError(
                "Spectral plotting is currently implemented "
                "only for surface k-space."
            )

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
        # H = flat_hk[ik, :, :]

        # E, V = np.linalg.eigh(H)

        G = 1/(1j*eta + omega[None,:] - E[ik, :,None])


        flat_A[ik,:] = -1 / np.pi * np.imag(np.einsum('iw, ai, ab, bi -> w',G, V[ik].conj(), total_proj, V[ik],  optimize=True))
        


    return flat_A.reshape(k_shape+(len(omega),))





