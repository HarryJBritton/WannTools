from wann_tools.wannham import WannHam

import numpy as np


# Load hr
ham = WannHam('data/in/PdBi2_hr.dat')

# make k space
k = np.zeros((10,10,10,3))

# FT  hr to hk
ham_k = ham.ft(k)
