import numpy as np


Space_Groups = {

    '139': {
        'G': np.array([0,0,0]),
        'Z': np.array([0,0,0.5]),
        'M': np.array([0.5,0,0]),
        'X': np.array([0.5,0,0]),
        'A': np.array([0.5,0.5,0.5]),
        'R': np.array([0.5,0,0.5])
    },

    '160' : {
        'G': np.array([0,0,0]),
        'K': np.array([1./3.,1./3.,0]),
        'M': np.array([0.5,0,0]),
        '<-Γ': 0.95* np.array([1./3,1./3.,0]),
        'Γ->': 0.95* np.array([1./3,1./3.,0]),
    },
}