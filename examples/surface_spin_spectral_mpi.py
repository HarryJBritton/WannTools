from mpi4py import MPI
import os

from wann_tools.kspace import KArea
from wann_tools.wannham import WannHam
from wann_tools.spectral import Spectral

import numpy as np
import gc


# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
FERMI = 8.0859948861
NL = 50
LP = 0
SP = 'x'
CHUNK_SIZE = 10
NK=20

VERBOSE = True

# TODO
# 1. load the nnkp 

# Load hr
if rank == 0 and VERBOSE:
    print('='*25)
    print('Loading hr')

# Completed for every rank
ham = WannHam('data/in/wannier90_hr.dat')

if rank == 0 and VERBOSE:
    print('='*25)
    print('building slab Hamiltonian...')

# Completed for every rank
ham.to_finite(
    layers=NL,
)

# Load nnkp
bvec =  np.array([
    [1.8688832,   0.0000000,   0.0000000],
    [0.0000000,  1.8688832,   0.0000000],
    [0.0000000,   0.0000000,   0.4839548]])


# ===================
# Build the k-space
# ===================

# make k space
if rank == 0 and VERBOSE:
    print('='*25)
    print('Constructing K structure...')

karea = KArea(bvec=bvec, num_per_vector=NK)
kgrid = karea.k

# Flatten only the k coordinates, not the Hamiltonians.
kpoints = kgrid.reshape(-1, 3)
n_k = len(kpoints)

omega = np.array([FERMI + 0.03])

# ===================
# Chunk the k-space
# ===================

local_indices = np.arange(rank, n_k, size)
local_A = np.empty((len(local_indices),), dtype=np.float64)

for batch_start in range(0, len(local_indices), CHUNK_SIZE):
    indices = local_indices[
        batch_start:batch_start + CHUNK_SIZE
    ]

    # get k
    k_chunk = kpoints[indices]
    # FT
    hk_chunk = ham.ft(k_chunk)

    print(
        f'rank {rank} of {size}: chunk {batch_start//CHUNK_SIZE} has size {hk_chunk.nbytes / (1024.0**3)} GB'
    )

    # get A
    A_chunk = Spectral(
        hk=hk_chunk,
        omega=omega,
        sproj=SP,
        lproj=LP,
        nl=NL,
        eta=0.01,
    )
    # Add to this rank's Spectral array
    local_A[batch_start:batch_start + len(indices)] = (
        np.asarray(A_chunk).reshape(len(indices), -1)[:, 0]
    )

    del hk_chunk, A_chunk, k_chunk
    gc.collect()

# Save only this rank's results and their global indices.
np.savez(
    f"data/tmp/spectral_rank_{rank}.npz",
    indices=local_indices,
    A=local_A,
)

comm.Barrier()

if rank == 0:

    # Reconstruct the map without gathering hk or eigenvectors.
    A_flat = np.empty(n_k, dtype=np.float64)

    for r in range(size):
        saved = np.load(f"data/tmp/spectral_rank_{r}.npz")
        A_flat[saved["indices"]] = saved["A"]
        saved.close()

    A = A_flat.reshape(kgrid.shape[:2])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.pcolormesh(
        kgrid[:, :, 0],
        kgrid[:, :, 1],
        A,
        cmap="bwr",
        vmin=-0.2,
        vmax=0.2,
    )
    ax.set_aspect("equal")
    plt.savefig("data/tmp/surface_spec.pdf")