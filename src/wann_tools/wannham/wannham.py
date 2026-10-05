import os
import numpy as np

class WannHam():

    # def __init__(self):
    #     self.file = None
    #     self.seedname = None
    #     self.nb = None
    #     self.nr = None
    #     self.ndeg = None

    def __init__(self, file):
        self.file = file
        self.load_hr()

    def load_hr(self):

        abs_path = os.getcwd() + '/' + self.file
        if not os.path.exists(abs_path):
            raise Exception(f'hr file at {abs_path} does not exist')
        else:
            print(f'Reading hr from {self.file}')

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
            self.hr = np.zeros((self.nr, self.nb, self.nb), dtype=complex)

            for ir in range(self.nr):
                for ia in range(self.nb):
                    for ib in range(self.nb):
                        row_data = f.readline().split()
                        self.rvec[ir, :] = row_data[:3]
                        self.hr[ir, int(row_data[3]) - 1,int(row_data[4]) - 1] = (
                            float(row_data[5]) + 1j * float(row_data[6])
                        )

    def ft(self, k):
        original_shape = k.shape
        flat_length = np.prod(original_shape[:-1])

        flat_k = k.reshape((flat_length,3))

        exp = np.exp(-1j * flat_k @ self.rvec.T)

        flat_hk = np.einsum('rab, kr -> kab', self.hr, exp)

        return flat_hk.reshape(original_shape[:-1] + (self.nb, self.nb,))