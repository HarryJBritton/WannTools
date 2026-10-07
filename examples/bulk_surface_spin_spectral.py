from wann_tools.kspace import KArea
from wann_tools.wannham import WannHam
from wann_tools.spectral import Spectral
import numpy as np
import matplotlib.pyplot as plt


FERMI = 8.0859948861

# TODO
# 1. load the nnkp file

# Load hr
ham = WannHam('data/in/wannier90_hr.dat')


# Load nnkp
bvec =  np.array([
    [1.8688832,   0.0000000,   0.0000000],
    [0.0000000,  1.8688832,   0.0000000],
    [0.0000000,   0.0000000,   0.4839548]])


# make k space
karea = KArea(
    bvec=bvec,
    num_per_vector=100
)

# FT  hr to hk
ham_k = ham.ft(karea.k)

eval = np.linalg.eigvalsh(ham_k) 

omega = np.array([FERMI+0.03])

A = Spectral(
    hk=ham_k,
    omega=omega,
    sproj='x',
    eta=0.01)

fig, ax = plt.subplots()


ax.pcolormesh(karea.k[:,:,0], karea.k[:,:,1], A[:,:,0], cmap='bwr', vmin=-.2, vmax=.2)
ax.set_aspect('equal')

# ax.set_xticks(
#     ticks=kpath.time_labels[0],
#     labels=kpath.time_labels[1]
# )

# for x in kpath.time_labels[0]:
#     ax.axvline(
#         x,
#         c='black',
#     )



plt.show()