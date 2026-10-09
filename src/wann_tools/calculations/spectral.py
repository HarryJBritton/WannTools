from wann_tools.calculations.base import ResourceEstimate

from .base import Calculation
from .registry import register_calculation

import numpy as np

@register_calculation
class SpectralCalculation(Calculation):

    pauli = {
        '0': np.array([[1,0],[0,1]], dtype=np.complex64),
        'x': np.array([[0,1],[1,0]], dtype=np.complex64),
        'y': np.array([[0,-1j],[1j,0]], dtype=np.complex64),
        'z': np.array([[1,0],[0,-1]], dtype=np.complex64),
    }

    def validate(self, config: dict, context) -> None:
        ...
    
    def estimate_resources(self, config: dict, context) -> ResourceEstimate:
        ...
    
    def run(self, config: dict, context, executor):
        ...


def Spectral(hk, omega, sproj=None, eta=0.05, lproj=None, nl=1):

    k_shape = hk.shape[:-2]
    nkp = np.prod(k_shape)
    nb = hk.shape[-1]
    no = nb // (2 * nl)

    # print(f'{nb} bands across {nl} layers\nwith {no} orbitals and {2} spins.')


    flat_hk = hk.reshape((nkp,nb,nb))

    # Spin projection
    if not sproj:
        ps = np.eye(2)
    elif sproj not in pauli.keys():
        raise Exception(f'spin projection {sproj} is not a Pauli matrix.\nMust be in {pauli.keys()}')
    else:
        ps = pauli[sproj]

    # TODO orbital projection
    po = np.eye(no)

    # Layer projection
    if not lproj:
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





