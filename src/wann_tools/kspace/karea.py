from typing import Any
import numpy as np

from wann_tools.kspace.base import KSpaceProvider
from .space_groups import Space_Groups

class KArea(KSpaceProvider):

   def __init__(
      self,
      bvec,
      group: str,
      n1: int,
      n2: int,
      level: float,
      fixed_axis: int,
      centre: str | None,
      width: float,
   ):

      # inherited properties
      self._bvec = bvec
      if group in Space_Groups.keys(): # Validate Space group
         self._group = Space_Groups[group]
      else:
         raise ValueError(f"Space group {group} not in the database")


      self.nk1 = n1
      self.nk2 = n2
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
      elif centre in self.space_group.keys():
         k = np.einsum('ji, j -> i', bvec,  Space_Groups['160'][centre], optimize=True) + T * self.width
      else:
         raise ValueError
      
      self._k = k

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
   def k(self) -> Any:
      return self._k




      



      

      
