from matplotlib import axes
import numpy as np

from .space_groups import Space_Groups

class KArea:

   def __init__(self, bvec, n1=20, n2=20, level: float=0, fixed_axis=2, centre=None, width=1.0):
      self.nk1 = n1
      self.nk2 = n2
      self.bvec = bvec
      self.nkps = self.nk1 * self.nk2
      self.width = width

      t1 = np.linspace(-0.5, 0.5, self.nk1)
      t2 = np.linspace(-0.5, 0.5, self.nk2)

      T1, T2 = np.meshgrid(t1,t2,indexing='xy')
      F = np.ones_like(T1) * level

      if fixed_axis == 0:
         T = np.dstack((F, T1, T2))
      elif fixed_axis == 1:
         T = np.dstack((T1, F, T2))
      elif fixed_axis == 2:
         T = np.dstack((T1, T2, F))
      else:
         raise Exception(f"axis {fixed_axis} not valid")


      if centre is None:
         k = np.einsum('ji, xyj -> xyi', bvec, T, optimize=True)
      elif centre in Space_Groups['160'].keys():
         k = np.einsum('ji, j -> i', bvec,  Space_Groups['160'][centre], optimize=True) + T * self.width
      else:
         raise ValueError
      
      self.k = k

   def mpi_split(self, rank, size):
      nkx, nky, ndim = self.k.shape

      # Number of k-grid rows assigned to each rank
      counts = np.full(size, nkx // size, dtype=int)
      counts[:nkx % size] += 1

      # Starting row for each rank
      starts = np.zeros(size, dtype=int)
      starts[1:] = np.cumsum(counts[:-1])

      local_start = starts[rank]
      local_count = counts[rank]

      if rank == 0:
         print('=' * 25)
         print(f'k-grid: {nkx} x {nky}')
         print(f'MPI ranks: {size}')


      print(
         f'Rank {rank}: '
         f'k rows {local_start}:{local_start + local_count}'
      )

      return self.k[local_start:local_start + local_count, :, :].copy(), counts

      



      

      
