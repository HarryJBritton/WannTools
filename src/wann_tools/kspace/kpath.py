import numpy as np

from .space_groups import Space_Groups


class KPath:

    def __init__(self, path, bvec, num_per_section=30, space_group='160'):
      self.nps = num_per_section
      self.bvec = bvec

      # Validate Space group
      if space_group in Space_Groups.keys():
        self.group = Space_Groups[space_group]
      else:
        raise Exception(f"Space group {space_group} not in the database")

      # Validate Path
      valid_path_flag = True
      for point in path:
         if not point in self.group.keys():
            valid_path_flag = False

      if valid_path_flag == False:
         raise Exception(f'Path {path} is not valid for space group {space_group}')
      else:
         self.path = path

      # Build path
      nkps = (len(self.path) - 1) * self.nps + 1
      k = np.zeros((nkps, 3), dtype=float)
      kl = np.zeros((nkps), dtype=float)

      for path_index in range(len(self.path)-1):
         k0 = self.bvec @ self.group[path[path_index]] 
         k1 = self.bvec @ self.group[path[path_index + 1]]  

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

      k[-1,:] = self.bvec @ self.group[path[-1]] 
      kl[-1] = np.linalg.norm( k[-1,:] -  k[-2,:]) + kl[-2]

      self.k = k
      self.length = kl
      self.time = kl / kl[-1]
      self.time_labels = ([self.time[i * self.nps] for i in range(len(self.path))],  [self.path[i]for i in range(len(self.path))])



      

      
