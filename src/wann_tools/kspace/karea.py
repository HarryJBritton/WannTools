from matplotlib import axes
import numpy as np

class KArea:

    def __init__(self, bvec, num_per_vector=20, level=0, fixed_axis=2):
      self.npv = num_per_vector
      self.bvec = bvec
      self.nkps = self.npv ** 2

      t = np.linspace(-0.5, 0.5, self.npv)

      T1, T2 = np.meshgrid(t,t,indexing='ij')
      F = np.ones_like(T1) * level

      if fixed_axis == 0:
         T = np.dstack((F, T1, T2))
      elif fixed_axis == 1:
         T = np.dstack((T1, F, T2))
      elif fixed_axis == 2:
         T = np.dstack((T1, T2, F))
      else:
         raise Exception(f"axis {fixed_axis} not valid")

      

      k = np.einsum('ji, xyj -> xyi', bvec, T)


      self.k = k



      

      
