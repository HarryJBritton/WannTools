import numpy as np
import pytest

from wann_tools.calculations.band import Band

def test_band():
    hk = np.array([[
        [1, +3j],
        [-3j, 1]]],
        dtype=complex,
    )

    # TODO fix this test

    # v, s, l = Band(
    #     hk=hk,
    #     ef=0,
    # )

    # assert v[0,0] == 1 - 3
    # assert v[0,1] == 1 + 3

    # assert s[0,0,0] == pytest.approx(0.0)
    # assert s[0,0,1] == pytest.approx(1.0)
    # assert s[0,0,2] == pytest.approx(0.0)

    # assert s[0,1,0] == pytest.approx(0.0)
    # assert s[0,1,1] == pytest.approx(-1.0)
    # assert s[0,1,2] == pytest.approx(0.0)