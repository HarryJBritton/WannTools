import numpy as np
from typing import Any

from wann_tools.kspace.base import KSpaceProvider
from .space_groups import Space_Groups



class KPath(KSpaceProvider):

   def __init__(
      self,
      bvec,
      group: str,
      path: list[str],
      num_per_section: int,
   ):

      # inherited properties
      self._bvec = bvec
      if group in Space_Groups.keys(): # Validate Space group
         self._group = Space_Groups[group]
      else:
         raise ValueError(f"Space group {group} not in the database")

      self.nps = num_per_section

      # Validate Path
      valid_path_flag = True
      for point in path:
         if not point in self.space_group.keys():
            valid_path_flag = False

      if valid_path_flag == False:
         raise Exception(f'Path {path} is not valid for space group {group}')
      else:
         self.path = path

      # Build path
      nkps = (len(self.path) - 1) * self.nps + 1
      k = np.zeros((nkps, 3), dtype=float)
      kl = np.zeros((nkps), dtype=float)

      for path_index in range(len(self.path)-1):
         k0 = np.einsum('ji, j -> i', bvec,self.space_group[path[path_index]], optimize=True )
         k1 = np.einsum('ji, j -> i', bvec, self.space_group[path[path_index + 1]], optimize=True  )

         for step in range(self.nps):
            # step along the k path
            k[step + self.nps * path_index, :] = k0 + (k1 - k0) * step / float(self.nps)

            if path_index == 0 and step == 0:
               old_k = k[step + self.nps * path_index, :]
               old_kl = 0
            else:
               old_k = k[step + self.nps * path_index - 1, :]
               old_kl = kl[step + self.nps * path_index - 1]

            k_diff = np.linalg.norm(k[step + self.nps * path_index, :] - old_k)
            kl[step + self.nps * path_index] = k_diff + old_kl

      # correct for the last high symmetry point

      k[-1,:] = np.einsum('ji, j -> i', bvec, self.space_group[path[-1]], optimize=True)
      kl[-1] = np.linalg.norm( k[-1,:] -  k[-2,:]) + kl[-2]

      self._k = k
      self.length = kl
      self.time = kl / kl[-1]
      self.time_labels = ([self.time[i * self.nps] for i in range(len(self.path))])


   @property
   def count(self) -> int:
      return np.prod(self.k.shape[:-1])

   @property
   def space_group(self) -> dict:
      return self._group

   @property
   def bvec(self) -> Any:
      return self._bvec
   
   @property
   def k(self)-> Any:
      return self._k

      

      
