import numpy as np

# TODO Do more Space groups (prehaps generalise this)
# TODO Better latex interpretation for Gamma
# TODO Implement a method of the towards concept

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
        'K\'': np.array([2./3., 2./3., 0. ]),
        '<-Γ': np.array([0.319191, 0.319191, 0.]),
        '<-M': np.array([0.353368, 0.293264, 0.]),
        'Γ->': np.array([0.313333, 0.313333, 0.]),
    },
}