from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import solve_ivp
from functools import partial



def SpeciesGrowth(spc, res, b, c, w, m):
    '''This function calculates the species growth rate
    Inputs:
    spc: species abundance, 1D array of length N
    res: resource level, 1D array of length R
    b: inverse timescale of species dynamics, 1D array of length N
    c: consumer-resource conversion matrix, 2D array of shape [N, R]
    w: value/efficiency of resource, 1D array of length R
    m: species maintenance cost, 1D array of length N
    Outputs: 
    g_spc: growth rate of species, 1D array of length N
    '''

    # Calculate the effective resource contribution for each species
    resource_contribution = np.dot(c, res * w)  # Matrix multiplication to get the sum(c_{i\beta} w_\beta R_\beta) for each species

    # Calculate the growth rate for each species
    g_spc = b * (resource_contribution - m)  # Element-wise operations based on the MacArthur model equation

    return g_spc


try:
    targets = process_hdf5_to_tuple('25.1', 3)
    target = targets[0]
    spc = np.array([1, 1])
    res = np.array([0.2, 0.4])
    b = np.array([1, 1])
    c = np.array([[0.4, 0.5], [0.9, 0.1]])
    w = np.array([0.8, 0.4])
    m = np.array([0.1, 0.05])
    assert np.allclose(SpeciesGrowth(spc, res, b, c, w, m), target)

    target = targets[1]
    spc = np.array([0.5, 0.6, 0.7])
    res = np.array([0.0, 0.0])
    b = np.array([0.1, 0.1, 0.1])
    c = np.array([[1, 0.1], [0.1, 1], [0.2, 0.3]])
    w = np.array([1, 1])
    m = np.array([0.05, 0.02, 0.1])
    assert np.allclose(SpeciesGrowth(spc, res, b, c, w, m), target)

    target = targets[2]
    spc = np.array([0.5, 0.6, 0.7])
    res = np.array([0.2, 0.4])
    b = np.array([0.1, 0.1, 0.1])
    c = np.array([[1, 0.1], [0.1, 1], [0.2, 0.3]])
    w = np.array([1, 1])
    m = np.array([0.05, 0.02, 0.1])
    assert np.allclose(SpeciesGrowth(spc, res, b, c, w, m), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e