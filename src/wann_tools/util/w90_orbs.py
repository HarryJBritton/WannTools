import numpy as np
from scipy.linalg import block_diag


def wannier90_angular_momentum(l):

    if not isinstance(l, (int, np.integer)) or l < 0:
        raise ValueError("l must be a non-negative integer")

    m = np.arange(-l, l+1)
    n = 2*l + 1

    # Complex basis 
    Lz = np.diag(m).astype(complex)

    Lp = np.zeros((n,n), dtype=complex)

    for col, mj in enumerate(m):
        if col < n - 1:
            Lp[col+1, col] = (
                np.sqrt(l*(l+1) - mj*(mj+1))
            )

    Lm = Lp.conj().T

    Lx = (Lp + Lm) / 2
    Ly = (Lp - Lm) / (2j)

     # Transformation to Wannier90 real harmonics
    U = np.zeros((n, n), dtype=complex)

    # First orbital: m=0
    U[l, 0] = 1.0

    col = 1

    for k in range(1, l + 1):
        phase = (-1)**k

        # cos(k phi)
        U[l-k, col] = 1 / np.sqrt(2)
        U[l+k, col] = phase / np.sqrt(2)
        col += 1

        # sin(k phi)
        U[l-k, col] = 1j / np.sqrt(2)
        U[l+k, col] = -1j * phase / np.sqrt(2)
        col += 1

    # Transform operators
    Lx = np.round(U.conj().T @ Lx @ U, 10)
    Ly = np.round(U.conj().T @ Ly @ U, 10)
    Lz = np.round(U.conj().T @ Lz @ U, 10)

    return Lx, Ly, Lz

def construct_am_op(
    orbital_ordering,
    include_spin=True
):
    orbs = {
        "s": 0,
        "p": 1,
        "d": 2,
        "f": 3,
    }
    
    if not all(orb in orbs for orb in orbital_ordering):
        raise ValueError("orbital_ordering must contain s, p, d or f")

    blocks = [
        wannier90_angular_momentum(orbs[orb])
        for orb in orbital_ordering
    ]

    Lx = block_diag(*[block[0] for block in blocks])
    Ly = block_diag(*[block[1] for block in blocks])
    Lz = block_diag(*[block[2] for block in blocks])

    if include_spin:
        I2 = np.eye(2, dtype=complex)

        Lx = np.kron(Lx, I2)
        Ly = np.kron(Ly, I2)
        Lz = np.kron(Lz, I2)

    return Lx, Ly, Lz