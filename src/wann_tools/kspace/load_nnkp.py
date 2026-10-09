import os
import numpy as np


def load_nnkp(file):
    abs_path = os.getcwd() + '/' + file
    if not os.path.exists(abs_path):
        raise Exception(f'nnkp file at {abs_path} does not exist')

    bvec = np.zeros((3,3), dtype=float)

    with open(abs_path) as f:

        while True: # Problematic

            line = f.readline()

            if not line:
                raise ValueError('No reciprocal lattice found')

            if 'begin recip_lattice' in line.strip():
                bvec[0,:] = f.readline().split()
                bvec[1,:] = f.readline().split()
                bvec[2,:] = f.readline().split()
                break
                

    return bvec