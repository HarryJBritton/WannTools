from wann_tools.wannham import WannHam
from wann_tools.kspace import KPath, load_nnkp
from wann_tools.spectral import Spectral
import numpy as np
import matplotlib.pyplot as plt


FERMI = 8.0859948861

# TODOload the nnkp file

# Load hr
ham = WannHam(
    'data/in/wannier90_hr.dat',
    np.complex64)


# Load nnkp
bvec =  load_nnkp('data/in/PdBi2.nnkp')


# make k space
kpath = KPath(
    path=['G', 'M', 'X', 'G'],
    bvec=bvec,
    space_group='139',
    num_per_section=100
)

# FT  hr to hk
ham_k = ham.ft(kpath.k)

eval = np.linalg.eigvalsh(ham_k) 

omega = np.linspace(-1, 1, 500) + FERMI

A = Spectral(
    hk=ham_k,
    omega=omega,
    sproj='x',
    eta=0.01)

fig, ax = plt.subplots()

T, O = np.meshgrid(kpath.time, omega - FERMI, indexing='ij')

ax.pcolormesh(T, O, A, cmap='bwr', vmin=-1, vmax=1)

ax.set_xticks(
    ticks=kpath.time_labels[0],
    labels=kpath.time_labels[1]
)

for x in kpath.time_labels[0]:
    ax.axvline(
        x,
        c='black',
    )

ax.set_xlim((0,1))

plt.show()