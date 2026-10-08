from mpi4py import MPI

from wann_tools.kspace import KArea
from wann_tools.wannham import WannHam
from wann_tools.spectral import Spectral

import numpy as np
import matplotlib.pyplot as plt

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
FERMI = 8.0859948861
NL = 30
LP = 0
SP = 'x'

# TODO
# 1. load the nnkp file

# Load hr
if rank == 0:
    print('='*25)
    print('Loading hr')

ham = WannHam('data/in/wannier90_hr.dat')

if rank == 0:
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
if rank == 0:
    print('='*25)
    print('Constructing K structure...')

karea = KArea(
    bvec=bvec,
    num_per_vector=100
)

local_k = karea.mpi_split(rank, size)

# ====
# FT
# ====

comm.Barrier() # All threads stop here until they are all here

if rank == 0:
    print('='*25)
    print('beginning Fourier transform...')


t0 = MPI.Wtime()

local_ham_k = ham.ft(local_k)

comm.Barrier()

if rank == 0:
    print(f'Fourier transform: {MPI.Wtime() - t0:.3f} s')

# =================
# Spectral Function
# =================

omega = np.array([FERMI+0.03])

local_A = Spectral(
    hk=local_ham_k,
    omega=omega,
    sproj=SP,
    lproj=LP,
    nl=NL,
    eta=0.01)

recv_counts = counts * nky * len(omega)
displs = np.zeros(size, dtype=int)
displs[1:] = np.cumsum(recv_counts[:-1])


if rank == 0:
    A_flat = np.empty(
        nkx * nky * len(omega),
        dtype=local_A.dtype,
    )
else:
    A_flat = None


comm.Gatherv(
    local_A.ravel(),
    [A_flat, recv_counts, displs, MPI.DOUBLE],
    root=0,
)

if rank == 0:

    A = A_flat.reshape(
        nkx,
        nky,
        len(omega),
    )

    print('='*25)
    print('Plotting...')
    fig, ax = plt.subplots()
    ax.pcolormesh(karea.k[:,:,0], karea.k[:,:,1], A[:,:,0], cmap='bwr', vmin=-.2, vmax=.2)
    ax.set_aspect('equal')

    plt.savefig('data/out/surface_spec.pdf')
