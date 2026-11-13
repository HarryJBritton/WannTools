
import numpy as np
import pytest

from wann_tools.util.w90_orbs import wannier90_angular_momentum


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_wannier90_angular_momentum(l):

    Lx, Ly, Lz = wannier90_angular_momentum(l)

    n = 2 * l + 1

    # Check matrix dimensions
    assert Lx.shape == (n, n)
    assert Ly.shape == (n, n)
    assert Lz.shape == (n, n)

    # Check Hermiticity
    assert np.allclose(Lx, Lx.conj().T)
    assert np.allclose(Ly, Ly.conj().T)
    assert np.allclose(Lz, Lz.conj().T)

    # Check angular momentum commutation relations (hbar = 1)
    assert np.allclose(Lx @ Ly - Ly @ Lx, 1j * Lz)
    assert np.allclose(Ly @ Lz - Lz @ Ly, 1j * Lx)
    assert np.allclose(Lz @ Lx - Lx @ Lz, 1j * Ly)

    # Check total angular momentum
    L_squared = Lx @ Lx + Ly @ Ly + Lz @ Lz

    expected = l * (l + 1) * np.eye(n)

    assert np.allclose(L_squared, expected)


def test_invalid_angular_momentum():

    with pytest.raises(ValueError):
        wannier90_angular_momentum(-1)

    with pytest.raises(ValueError):
        wannier90_angular_momentum(1.5)
