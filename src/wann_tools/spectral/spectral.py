import numpy as np


pauli = {
    '0': np.array([[1,0],[0,1]], dtype=complex),
    'x': np.array([[0,1],[1,0]], dtype=complex),
    'y': np.array([[0,-1j],[1j,0]], dtype=complex),
    'z': np.array([[1,0],[0,-1]], dtype=complex),
}

def Spectral(hk, omega, sproj=None, eta=0.05):

    k_shape = hk.shape[:-2]
    nkp = np.prod(k_shape)
    nb = hk.shape[-1]

    flat_hk = hk.reshape((nkp,nb,nb))

    # Spin projection
    if not sproj:
        ps = np.eye(2)
    elif sproj not in pauli.keys():
        raise Exception(f'spin projection {sproj} is not a Pauli matrix.\nMust be in {pauli.keys()}')
    else:
        ps = pauli[sproj]


    # TODO orbital projection
    po = np.eye(nb//2)

    # TODO layer projection
    pl = np.eye(1)

    total_proj = np.kron(pl, np.kron(po, ps))

    flat_A = np.zeros((nkp, len(omega)))
    for ik in range(nkp):
        H = flat_hk[ik, :, :]

        E, V = np.linalg.eigh(H)

        G = 1/(1j*eta + omega[None,:] - E[:,None])

        flat_A[ik,:] = -1 / np.pi * np.imag(np.einsum('iw, ai, ab, bi -> w',G, V.conj(), total_proj, V))
        


    return flat_A.reshape(k_shape+(len(omega),))





