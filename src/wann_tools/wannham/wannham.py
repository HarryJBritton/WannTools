import os
import numpy as np

class WannHam():

    # def __init__(self):
    #     self.file = None
    #     self.seedname = None
    #     self.nb = None
    #     self.nr = None
    #     self.ndeg = None

    def __init__(self, hr_file, nnkp_file, dtype):
        self.hr_file = hr_file
        self.nnkp_file = nnkp_file

        self.dtype=dtype
        self.load_hr()
        self.load_nnkp()

    def load_hr(self):

        abs_path = os.getcwd() + '/' + self.hr_file
        if not os.path.exists(abs_path):
            raise Exception(f'hr file at {abs_path} does not exist')
        # else:
        #     print(f'Reading hr from {self.file}')

        with open(abs_path) as f:
            f.readline() # read meta data
            self.nb = int(f.readline())
            self.nr = int(f.readline())

            deg_lines = int(np.ceil(self.nr / 15.0))
            self.ndeg = np.zeros(self.nr, dtype=int)
            for il in range(deg_lines):
                ndeg_line = list(f.readline().split())
                for jl in range(len(ndeg_line)):
                    self.ndeg[15 * il + jl] = ndeg_line[jl]

            self.rvec = np.zeros((self.nr,3), dtype=float)
            self.hr = np.zeros((self.nr, self.nb, self.nb), dtype=self.dtype)

            for ir in range(self.nr):
                for ia in range(self.nb):
                    for ib in range(self.nb):
                        row_data = f.readline().split()
                        self.rvec[ir, :] = row_data[:3]
                        self.hr[ir, int(row_data[3]) - 1,int(row_data[4]) - 1] = (
                            float(row_data[5]) + 1j * float(row_data[6])
                        )

    def load_nnkp(self):
        abs_path = os.getcwd() + '/' + self.nnkp_file
        if not os.path.exists(abs_path):
            raise Exception(f'nnkp file at {abs_path} does not exist')

        lvec = np.zeros((3,3), dtype=float)
        
        with open(abs_path) as f:
    
            while True: # Problematic
    
                line = f.readline()
    
                if not line:
                    raise ValueError('No real lattice found')
    
                if 'begin real_lattice' in line.strip():
                    lvec[0,:] = f.readline().split()
                    lvec[1,:] = f.readline().split()
                    lvec[2,:] = f.readline().split()
                    break

        self.lvec = lvec
                        

    def ft(self, k):
        original_shape = k.shape
        flat_length = np.prod(original_shape[:-1])

        flat_k = k.reshape((flat_length,3))

        phase = np.einsum('ki, rj, ji -> kr', flat_k, self.rvec,  self.lvec)

        exp = np.exp(-1j * phase)

        flat_hk = np.einsum('rab, kr, r -> kab', self.hr, exp, 1 / self.ndeg, optimize=True)

        return flat_hk.reshape(original_shape[:-1] + (self.nb, self.nb,))

    def to_finite(self, layers = 5, axes = 2):

        r_plane = self.rvec[:, :2]

        u, idx = np.unique(r_plane, axis=0, return_inverse=True)

        new_hr = np.zeros((u.shape[0], self.nb * layers, self.nb * layers), dtype=self.dtype)
        new_rvec = np.hstack((u, np.zeros((u.shape[0], 1))))

        new_ndeg = np.zeros((u.shape[0]))

        for ri in range(len(idx)):
            hr_idx = idx[ri]

            z = self.rvec[ri][2]

            if z < layers and z > -layers:
                new_hr[hr_idx, :, :] += np.kron(
                    np.eye(layers,k=int(z)),
                    self.hr[ri, :, :] 
                )
            
            if z == 0:
                new_ndeg[hr_idx] = self.ndeg[ri]

            
        
        self.nb = self.nb * layers
        self.nr = new_rvec.shape[0]
        self.rvec = new_rvec
        self.ndeg = new_ndeg

        self.hr = new_hr

