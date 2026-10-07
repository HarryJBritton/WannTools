from wann_tools.kspace import KArea
from wann_tools.wannham import WannHam
from wann_tools.spectral import Spectral
import numpy as np
import matplotlib.pyplot as plt


FERMI = 8.0859948861
NL =20

# TODO
# 1. load the nnkp file

# Load hr
ham = WannHam('data/in/wannier90_hr.dat')

print('='*25)
print('building slab Hamiltonian...')
ham.to_finite(
    layers=NL,
)

# Load nnkp
bvec =  np.array([
    [1.8688832,   0.0000000,   0.0000000],
    [0.0000000,  1.8688832,   0.0000000],
    [0.0000000,   0.0000000,   0.4839548]])


# make k space
print('='*25)
print('Constructing K structure...')
karea = KArea(
    bvec=bvec,
    num_per_vector=100
)

# FT  hr to hk
print('='*25)
print('beginning Fourier transform...')
ham_k = ham.ft(karea.k)

omega = np.array([FERMI+0.03])

print('='*25)
print('Calculating spectral function...')
A = Spectral(
    hk=ham_k,
    omega=omega,
    sproj='x',
    lproj=0,
    nl=NL,
    eta=0.01)


print('='*25)
print('Plotting...')
fig, ax = plt.subplots()
ax.pcolormesh(karea.k[:,:,0], karea.k[:,:,1], A[:,:,0], cmap='bwr', vmin=-.2, vmax=.2)
ax.set_aspect('equal')

plt.show()